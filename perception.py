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

_ollama_client = ollama.Client(host=OLLAMA_HOST)

# 最新干净帧（供 observe_camera 工具实时调用）
_latest_raw_frame: np.ndarray | None = None
_stop_event = threading.Event()  # 程序退出时 set，让 observe_camera 的 sleep 提前返回


def get_latest_frame() -> np.ndarray | None:
    """返回感知循环最近捕获的原始帧，供 ReAct observe_camera 工具使用。"""
    return _latest_raw_frame

# 手部骨架连线（21 个关键点之间的父子关系）
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm knuckle arch
]

# MediaPipe 手势名称 -> 可读标签
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


# ── 置信度计算 ────────────────────────────────────────────────

def _pose_confidence(landmarks: list) -> float:
    """用鼻子/双肩/双髋这5个关键点的平均 visibility 作为检测置信度"""
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


# ── 绘制工具 ──────────────────────────────────────────────────

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
    """绘制 21 个手部关键点 + 骨架连线 + 手势标签"""
    fh, fw = frame.shape[:2]
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_landmarks]

    # 连线
    for a, b in _HAND_CONNECTIONS:
        if 0 <= pts[a][0] < fw and 0 <= pts[a][1] < fh and \
           0 <= pts[b][0] < fw and 0 <= pts[b][1] < fh:
            cv2.line(frame, pts[a], pts[b], HAND_LINE_COLOR, 1, cv2.LINE_AA)

    # 关键点
    for i, (px, py) in enumerate(pts):
        if 0 <= px < fw and 0 <= py < fh:
            # 指尖（4/8/12/16/20）画稍大的实心圆
            r = HAND_DOT_RADIUS + 2 if i in (4, 8, 12, 16, 20) else HAND_DOT_RADIUS
            cv2.circle(frame, (px, py), r, HAND_DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), r, (0, 0, 0), 1, cv2.LINE_AA)  # 黑边

    # 手势标签（腕关节上方）
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


# ── Moondream2 推理 ───────────────────────────────────────────

def query_moondream(frame_bgr: np.ndarray) -> str:
    resized = cv2.resize(frame_bgr, (640, 480))
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(tmp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    try:
        r = _ollama_client.generate(model=MOONDREAM_MODEL, prompt=MOONDREAM_PROMPT,
                                    images=[tmp_path])
        text = r.response.strip()
        if not text:
            return "no activity"
        return text.split("\n")[0].strip()
    except Exception as e:
        console.print(f"{LOG_A} moondream error: {e}")
        return "moondream unavailable"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── 本地行为分类（qwen2.5:1.5b 小脑） ────────────────────────
# 职责拆分：
#   qwen  → 只判断行为是否健康（最简单的分类任务）
#   Python → 时间豁免规则（可靠，零成本）
#   合并   → 得出 should_escalate

# qwen 采用 yes/no 关键词检测，疑罪从无，默认 healthy
# 只有描述中明确出现以下娱乐关键词才判 unhealthy
# {context_section} 由 _qwen_health_check 动态填入（有上下文时注入，无则空字符串）
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

# Python 层关键词预过滤（超明显情况直接截断，不走 LLM）
_UNHEALTHY_KEYWORDS = [
    'scrolling', 'scroll', 'social media', 'tiktok', 'instagram',
    'selfie', 'taking a selfie',
    'watching television', 'watching tv', 'watching a show', 'watching a movie',
    'watching video', 'streaming', 'netflix',
    'playing video game', 'playing a game', 'gaming',
    'lying in bed', 'lying on bed',
]

def classify_behavior(vision_text: str, context: str = "") -> tuple[bool, bool]:
    """返回 (is_healthy, should_escalate)，全天候无豁免。"""
    is_healthy = _qwen_health_check(vision_text, context)
    should_escalate = not is_healthy

    console.print(
        f"{LOG_A} cerebellum → healthy={'yes' if is_healthy else 'no'} "
        f"escalate={'yes' if should_escalate else 'no'}"
    )
    return is_healthy, should_escalate


def _qwen_health_check(vision_text: str, context: str = "") -> bool:
    """qwen2.5:1.5b 结合近期上下文做行为健康判断，返回 True=healthy。
    策略：疑罪从无 — 只有明确出现摆烂关键词才判 unhealthy。
    """
    text_lower = vision_text.lower()

    # Python 预过滤：超明显关键词直接短路，不走 LLM
    if any(kw in text_lower for kw in _UNHEALTHY_KEYWORDS):
        console.print(f"{LOG_A} qwen → keyword match → unhealthy")
        return False

    try:
        context_section = (
            f"Recent context (use this to adjust your judgment):\n{context}\n\n"
            if context else ""
        )
        prompt = _CLASSIFIER_PROMPT.format(context_section=context_section, text=vision_text)
        r = _ollama_client.generate(model=LOCAL_CLASSIFIER_MODEL, prompt=prompt)
        raw = r.response.strip().lower() if r.response.strip() else "no"
        # 在响应中找第一个 yes 或 no
        for word in raw.split():
            word = word.rstrip('.,:')
            if word == 'yes':
                console.print(f"{LOG_A} qwen → yes → unhealthy")
                return False
            if word == 'no':
                console.print(f"{LOG_A} qwen → no → healthy")
                return True
        # 没找到 yes/no：默认 healthy（疑罪从无）
        console.print(f"{LOG_A} qwen → unclear(\"{raw[:20]}\") → healthy (default)")
        return True
    except Exception as e:
        console.print(f"{LOG_A} qwen error: {e}")
        return True  # 出错也默认 healthy，避免误报


# ── 主感知循环 ────────────────────────────────────────────────

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

    # Single-element lists used as mutable containers so the _analyze thread
    # can write back values visible to the main loop (Python closure limitation).
    last_t      = [0.0]                  # Timestamp of last capture dispatch
    behavior    = ["waiting for scan..."] # Latest Moondream behavior description
    lock        = threading.Lock()
    busy        = [False]                # True while _analyze thread is running
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
            console.print(f"{LOG_A} [{ts}] moondream={moondream_elapsed:.1f}s -> \"{text}\"")
            if state_callback:
                state_callback(text, ts, is_healthy, should_escalate)
        except Exception as e:
            console.print(f"{LOG_A} analyze error: {e}")
        finally:
            with lock:
                busy[0] = False  # Always reset so the main loop never gets stuck waiting

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
        _gest_ok = True  # 手势识别器出错后禁用，防止崩溃

        while True:
            ret, frame = cap.read()
            if not ret:
                console.print(f"{LOG_A} camera read failed")
                break

            raw_frame = frame.copy()   # 干净帧，发给 moondream / observe_camera
            global _latest_raw_frame
            _latest_raw_frame = raw_frame
            fh, fw    = frame.shape[:2]
            now       = time.time()
            ts_ms     = int(now * 1000) - start_ms

            # fps
            frame_count[0] += 1
            dt = now - fps_t[0]
            if dt >= 1.0:
                fps_val[0] = frame_count[0] / dt
                frame_count[0] = 0
                fps_t[0] = now

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            # ── Pose 检测 ─────────────────────────────────
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

                # "PERSON 94%" 标签
                header = f"PERSON  {conf*100:.0f}%"
                cv2.putText(frame, header, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            GREEN_BOX_COLOR, 1, cv2.LINE_AA)

                # 行为描述（绿框右侧）
                with lock:
                    beh = behavior[0]
                avail = max(100, fw - min(x2+10, fw-10) - 8)
                _draw_label(frame, beh, min(x2+10, fw-10), y1+22, max_width=avail)

            # ── 手势检测 ──────────────────────────────────
            if _gest_ok:
                try:
                    gest_res = gest_det.recognize_for_video(mp_img, ts_ms)
                except RuntimeError as e:
                    console.print(f"{LOG_A} gesture error (disabled): {e}")
                    _gest_ok = False
                    gest_res = None
            else:
                gest_res = None

            for hi, hand_lms in enumerate(gest_res.hand_landmarks if gest_res else []):
                # 手势名称
                gesture_name = "None"
                if hi < len(gest_res.gestures) and gest_res.gestures[hi]:
                    gesture_name = gest_res.gestures[hi][0].category_name

                # 手的左右
                handedness = "Right"
                if hi < len(gest_res.handedness) and gest_res.handedness[hi]:
                    handedness = gest_res.handedness[hi][0].category_name

                _draw_hand(frame, hand_lms, gesture_name, handedness)

            # ── 定时触发 moondream ────────────────────────
            if now - last_t[0] >= CAPTURE_INTERVAL_SEC and not busy[0]:
                with lock:
                    busy[0] = True
                last_t[0] = now
                threading.Thread(target=_analyze, args=(raw_frame,), daemon=True).start()

            _draw_hud(frame, now, last_t[0], person_detected, busy[0], fps_val[0])
            cv2.imshow("Cyber-Superego", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                console.print(f"{LOG_A} quit")
                _stop_event.set()  # 通知 observe_camera 的 sleep 提前退出
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_perception_loop()
