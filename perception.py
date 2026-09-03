"""
perception.py — Node A: local edge perception
camera -> mediapipe pose+gesture -> moondream2 -> annotated display
"""
import os
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import ollama
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)
from rich.console import Console

from config import (
    CAMERA_INDEX,
    CAPTURE_INTERVAL_SEC,
    MEDIAPIPE_CONFIDENCE,
    GREEN_BOX_COLOR,
    GREEN_BOX_THICKNESS,
    TEXT_COLOR,
    TEXT_FONT_SCALE,
    TEXT_THICKNESS,
    PERSON_BOX_PADDING,
    HAND_DOT_COLOR,
    HAND_LINE_COLOR,
    HAND_DOT_RADIUS,
    GESTURE_COLOR,
    MOONDREAM_MODEL,
    MOONDREAM_PROMPT,
    LOCAL_CLASSIFIER_MODEL,
    OLLAMA_HOST,
    LOG_A,
)

console = Console()

_POSE_MODEL  = Path(__file__).parent / "pose_landmarker_lite.task"
_GESTURE_MODEL = Path(__file__).parent / "gesture_recognizer.task"
_MAX_DESCRIPTION_CHARS = 1000
_MAX_CONTEXT_CHARS = 2000
_BIDI_CONTROLS = {chr(code) for code in (0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069)}

_ollama_client = ollama.Client(host=OLLAMA_HOST)

_latest_raw_frame: np.ndarray | None = None
_stop_event = threading.Event()


def _clean_text(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = " ".join(value.split())
    text = "".join(ch for ch in text if ch not in _BIDI_CONTROLS and ord(ch) >= 32 and ord(ch) != 127).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _validate_frame(frame: object) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ValueError("camera frame must be a numpy array")
    if frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError("camera frame has invalid dimensions")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError("camera frame has unsupported channel count")
    return frame


def get_latest_frame() -> np.ndarray | None:
    """Return an isolated copy of the latest clean frame for observe_camera."""
    frame = _latest_raw_frame
    return None if frame is None else frame.copy()


_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

_GESTURE_LABEL = {
    "None":         "",
    "Closed_Fist":  "FIST",
    "Open_Palm":    "PALM",
    "Pointing_Up":  "POINT",
    "Thumb_Down":   "THUMB-",
    "Thumb_Up":     "THUMB+",
    "Victory":      "PEACE",
    "ILoveYou":     "ILY",
}


def _pose_confidence(landmarks: list) -> float:
    key_ids = {0, 11, 12, 23, 24}
    vals = [lm.visibility for i, lm in enumerate(landmarks) if i in key_ids]
    return sum(vals) / len(vals) if vals else 0.0


def _get_person_bbox(
    landmarks: list, frame_h: int, frame_w: int
) -> tuple[int, int, int, int] | None:
    xs = [lm.x * frame_w for lm in landmarks if lm.visibility > MEDIAPIPE_CONFIDENCE]
    ys = [lm.y * frame_h for lm in landmarks if lm.visibility > MEDIAPIPE_CONFIDENCE]
    if not xs:
        return None
    x1 = max(0, int(min(xs)) - PERSON_BOX_PADDING)
    y1 = max(0, int(min(ys)) - PERSON_BOX_PADDING)
    x2 = min(frame_w, int(max(xs)) + PERSON_BOX_PADDING)
    y2 = min(frame_h, int(max(ys)) + PERSON_BOX_PADDING)
    return x1, y1, x2, y2


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, max_width: int = 380,
                color: tuple = None) -> None:
    if not text:
        return
    color = color or TEXT_COLOR
    font   = cv2.FONT_HERSHEY_SIMPLEX
    lh     = int(TEXT_FONT_SCALE * 38 + 8)
    mc     = max(8, max_width // max(1, int(TEXT_FONT_SCALE * 16)))
    lines  = _wrap_text(text, mc)
    bg_w   = max(cv2.getTextSize(l, font, TEXT_FONT_SCALE, TEXT_THICKNESS)[0][0] for l in lines) + 14
    bg_h   = lh * len(lines) + 10
    fh, fw = frame.shape[:2]
    if x + bg_w > fw: x = max(0, fw - bg_w - 4)
    if y + bg_h > fh: y = max(0, fh - bg_h - 4)
    ov = frame.copy()
    cv2.rectangle(ov, (x-4, y-lh+4), (x+bg_w, y+bg_h-lh+4), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i*lh), font, TEXT_FONT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA)


def _draw_hand(frame: np.ndarray, hand_landmarks: list, gesture: str, handedness: str) -> None:
    fh, fw = frame.shape[:2]
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_landmarks]
    for a, b in _HAND_CONNECTIONS:
        if 0 <= pts[a][0] < fw and 0 <= pts[a][1] < fh and \
           0 <= pts[b][0] < fw and 0 <= pts[b][1] < fh:
            cv2.line(frame, pts[a], pts[b], HAND_LINE_COLOR, 1, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        if 0 <= px < fw and 0 <= py < fh:
            r = HAND_DOT_RADIUS + 2 if i in (4, 8, 12, 16, 20) else HAND_DOT_RADIUS
            cv2.circle(frame, (px, py), r, HAND_DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), r, (0, 0, 0), 1, cv2.LINE_AA)
    label = _GESTURE_LABEL.get(gesture, gesture)
    if label:
        wx, wy = pts[0]
        side = "L" if handedness == "Left" else "R"
        tag = f"[{side}] {label}"
        cv2.putText(frame, tag, (wx - 20, max(0, wy - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GESTURE_COLOR, 2, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, now: float, last_t: float,
              person: bool, analyzing: bool, fps: float) -> None:
    fh, fw = frame.shape[:2]
    if analyzing:
        scan = "scanning..."
    else:
        nxt = max(0, CAPTURE_INTERVAL_SEC - (now - last_t))
        scan = f"next_scan={nxt:.0f}s"
    lines = [
        "CYBER-SUPEREGO v0.1",
        datetime.now().strftime("%H:%M:%S"),
        f"fps={fps:.1f}  person={'1' if person else '0'}",
        scan,
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    lh, pad = 20, 8
    bg_x1 = fw - 230
    bg_y1 = fh - len(lines)*lh - pad*2
    ov = frame.copy()
    cv2.rectangle(ov, (bg_x1, bg_y1), (fw-4, fh-4), (0,0,0), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        c = (0,255,0) if i == 0 else (180,180,180)
        cv2.putText(frame, line, (bg_x1+pad, bg_y1+pad+lh+i*lh),
                    font, 0.42, c, 1, cv2.LINE_AA)


def query_moondream(frame_bgr: np.ndarray) -> str:
    frame_bgr = _validate_frame(frame_bgr)
    resized = cv2.resize(frame_bgr, (640, 480))
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        if not cv2.imwrite(tmp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError("camera frame encoding failed")
        response = _ollama_client.generate(
            model=MOONDREAM_MODEL,
            prompt=MOONDREAM_PROMPT,
            images=[tmp_path],
        )
        raw = getattr(response, "response", "")
        if not isinstance(raw, str) or not raw.strip():
            return "no activity"
        return _clean_text(raw, field="camera description", limit=_MAX_DESCRIPTION_CHARS)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"{LOG_A} moondream unavailable ({exc.__class__.__name__})")
        return "moondream unavailable"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


_CLASSIFIER_PROMPT = """Does this description contain any of these activities?
- scrolling phone / social media / TikTok / Instagram
- taking selfie / posing in mirror / phone camera
- watching TV / television / a show / a movie / streaming / Netflix / video on screen
- playing video game / gaming / console
- lying in bed on phone

{context_section}If NONE of the above are mentioned, answer no.
If any of the above are clearly present, answer yes.

Description: {text}
Answer (yes or no):"""

_UNHEALTHY_KEYWORDS = [
    'scrolling', 'scroll', 'social media', 'tiktok', 'instagram',
    'selfie', 'taking a selfie',
    'watching television', 'watching tv', 'watching a show', 'watching a movie',
    'watching video', 'streaming', 'netflix',
    'playing video game', 'playing a game', 'gaming',
    'lying in bed', 'lying on bed',
]


def classify_behavior(vision_text: str, context: str = "") -> tuple[bool, bool]:
    is_healthy = _qwen_health_check(vision_text, context)
    should_escalate = not is_healthy
    console.print(
        f"{LOG_A} cerebellum → healthy={'yes' if is_healthy else 'no'} "
        f"escalate={'yes' if should_escalate else 'no'}"
    )
    return is_healthy, should_escalate


def _qwen_health_check(vision_text: str, context: str = "") -> bool:
    vision_text = _clean_text(vision_text, field="vision text", limit=_MAX_DESCRIPTION_CHARS)
    if context:
        context = _clean_text(context, field="classifier context", limit=_MAX_CONTEXT_CHARS)
    text_lower = vision_text.lower()
    if any(kw in text_lower for kw in _UNHEALTHY_KEYWORDS):
        console.print(f"{LOG_A} qwen → keyword match → unhealthy")
        return False
    try:
        context_section = (
            f"Recent context (use this to adjust your judgment):\n{context}\n\n"
            if context else ""
        )
        prompt = _CLASSIFIER_PROMPT.format(context_section=context_section, text=vision_text)
        response = _ollama_client.generate(model=LOCAL_CLASSIFIER_MODEL, prompt=prompt)
        value = getattr(response, "response", "")
        raw = value.strip().lower() if isinstance(value, str) and value.strip() else "no"
        for word in raw.split():
            word = word.rstrip('.,:')
            if word == 'yes':
                console.print(f"{LOG_A} qwen → yes → unhealthy")
                return False
            if word == 'no':
                console.print(f"{LOG_A} qwen → no → healthy")
                return True
        console.print(f"{LOG_A} qwen → unclear response → healthy (default)")
        return True
    except Exception as exc:  # noqa: BLE001
        console.print(f"{LOG_A} qwen unavailable ({exc.__class__.__name__})")
        return True


def run_perception_loop(state_callback=None, get_context=None):
    for p, name in [(_POSE_MODEL, "pose_landmarker_lite.task"),
                    (_GESTURE_MODEL, "gesture_recognizer.task")]:
        if not p.exists():
            console.print(f"{LOG_A} missing model: {name}")
            return

    console.print(f"{LOG_A} cam={CAMERA_INDEX} interval={CAPTURE_INTERVAL_SEC}s  q/ESC to quit")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        console.print(f"{LOG_A} cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_t      = [0.0]
    behavior    = ["waiting for scan..."]
    lock        = threading.Lock()
    busy        = [False]
    frame_count = [0]
    fps_t       = [time.time()]
    fps_val     = [0.0]

    def _analyze(snap: np.ndarray):
        t0 = time.time()
        try:
            console.print(f"{LOG_A} dispatching moondream")
            text = query_moondream(snap)
            moondream_elapsed = time.time() - t0
            context = get_context() if get_context else ""
            is_healthy, should_escalate = classify_behavior(text, context)
            ts = datetime.now().strftime("%H:%M:%S")
            with lock:
                behavior[0] = text
            console.print(f"{LOG_A} [{ts}] moondream={moondream_elapsed:.1f}s -> description ready")
            if state_callback:
                state_callback(text, ts, is_healthy, should_escalate)
        except Exception as exc:  # noqa: BLE001
            console.print(f"{LOG_A} analyze failed ({exc.__class__.__name__})")
        finally:
            with lock:
                busy[0] = False

    pose_opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MEDIAPIPE_CONFIDENCE,
        min_pose_presence_confidence=MEDIAPIPE_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_CONFIDENCE,
    )
    gest_opts = GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_GESTURE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MEDIAPIPE_CONFIDENCE,
        min_hand_presence_confidence=MEDIAPIPE_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_CONFIDENCE,
    )

    with PoseLandmarker.create_from_options(pose_opts) as pose_det, \
         GestureRecognizer.create_from_options(gest_opts) as gest_det:
        console.print(f"{LOG_A} mediapipe ready  (pose + gesture)")
        start_ms = int(time.time() * 1000)
        _gest_ok = True

        while True:
            ret, frame = cap.read()
            if not ret:
                console.print(f"{LOG_A} camera read failed")
                break
            raw_frame = frame.copy()
            global _latest_raw_frame
            _latest_raw_frame = raw_frame
            fh, fw    = frame.shape[:2]
            now       = time.time()
            ts_ms     = int(now * 1000) - start_ms
            frame_count[0] += 1
            dt = now - fps_t[0]
            if dt >= 1.0:
                fps_val[0] = frame_count[0] / dt
                frame_count[0] = 0
                fps_t[0] = now

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            pose_res = pose_det.detect_for_video(mp_img, ts_ms)
            person_detected = False
            for pose_lms in pose_res.pose_landmarks:
                bbox = _get_person_bbox(pose_lms, fh, fw)
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                conf = _pose_confidence(pose_lms)
                person_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN_BOX_COLOR, GREEN_BOX_THICKNESS)
                header = f"PERSON  {conf*100:.0f}%"
                cv2.putText(frame, header, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            GREEN_BOX_COLOR, 1, cv2.LINE_AA)
                with lock:
                    beh = behavior[0]
                avail = max(100, fw - min(x2+10, fw-10) - 8)
                _draw_label(frame, beh, min(x2+10, fw-10), y1+22, max_width=avail)

            if _gest_ok:
                try:
                    gest_res = gest_det.recognize_for_video(mp_img, ts_ms)
                except RuntimeError as exc:
                    console.print(f"{LOG_A} gesture unavailable ({exc.__class__.__name__})")
                    _gest_ok = False
                    gest_res = None
            else:
                gest_res = None

            for hi, hand_lms in enumerate(gest_res.hand_landmarks if gest_res else []):
                gesture_name = "None"
                if hi < len(gest_res.gestures) and gest_res.gestures[hi]:
                    gesture_name = gest_res.gestures[hi][0].category_name
                handedness = "Right"
                if hi < len(gest_res.handedness) and gest_res.handedness[hi]:
                    handedness = gest_res.handedness[hi][0].category_name
                _draw_hand(frame, hand_lms, gesture_name, handedness)

            if now - last_t[0] >= CAPTURE_INTERVAL_SEC and not busy[0]:
                with lock:
                    busy[0] = True
                last_t[0] = now
                threading.Thread(target=_analyze, args=(raw_frame,), daemon=True).start()

            _draw_hud(frame, now, last_t[0], person_detected, busy[0], fps_val[0])
            cv2.imshow("Cyber-Superego", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                console.print(f"{LOG_A} quit")
                _stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_perception_loop()
