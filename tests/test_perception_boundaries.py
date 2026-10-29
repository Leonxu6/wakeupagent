import numpy as np
import pytest

import perception


def test_video_timestamps_use_elapsed_monotonic_time():
    assert perception._next_video_timestamp_ms(10.0, -1, now=10.125) == 125


def test_video_timestamps_remain_strictly_increasing_within_one_millisecond():
    assert perception._next_video_timestamp_ms(10.0, 125, now=10.1254) == 126


def test_video_timestamps_do_not_move_backwards():
    assert perception._next_video_timestamp_ms(10.0, 125, now=9.5) == 126


def test_latest_frame_returns_defensive_copy(monkeypatch):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", original)
    copy = perception.get_latest_frame()
    assert copy is not original
    copy[0, 0, 0] = 255
    assert original[0, 0, 0] == 0


def test_storing_latest_frame_takes_ownership_of_its_buffer(monkeypatch):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", None)

    perception._store_latest_frame(original)
    original[0, 0, 0] = 255

    stored = perception.get_latest_frame()
    assert stored is not None
    assert stored[0, 0, 0] == 0


def test_storing_invalid_frame_preserves_the_previous_snapshot(monkeypatch):
    previous = np.ones((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", previous)

    with pytest.raises(ValueError, match="camera frame"):
        perception._store_latest_frame(np.zeros((2, 2, 5), dtype=np.uint8))

    assert perception._latest_raw_frame is previous


def test_latest_frame_can_require_a_capture_after_request(monkeypatch):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", original)
    monkeypatch.setattr(perception, "_latest_frame_captured_at", 10.0)

    assert perception.get_latest_frame(captured_after=10.1) is None
    assert perception.get_latest_frame(captured_after=10.0) is not None


def test_frame_wait_returns_an_already_fresh_defensive_copy(monkeypatch):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", original)
    monkeypatch.setattr(perception, "_latest_frame_captured_at", 10.0)
    perception._stop_event.clear()

    result = perception.wait_for_latest_frame(captured_after=9.0, timeout=1.0)

    assert result is not None
    assert result is not original


@pytest.mark.parametrize("timeout", [True, 0, -1, 3601, "1", float("nan"), float("inf")])
def test_frame_wait_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="frame wait timeout"):
        perception.wait_for_latest_frame(captured_after=1.0, timeout=timeout)


@pytest.mark.parametrize("value", [True, "10", float("nan"), float("inf")])
def test_latest_frame_rejects_invalid_capture_threshold(value):
    with pytest.raises(ValueError, match="finite monotonic"):
        perception.get_latest_frame(captured_after=value)


@pytest.mark.parametrize("frame", [None, object(), np.array([]), np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 5))])
def test_validate_frame_rejects_invalid_shapes(frame):
    with pytest.raises(ValueError):
        perception._validate_frame(frame)


def test_clean_text_normalizes_controls_and_bounds_output():
    text = perception._clean_text("  person\nreading\u202etext  ", field="camera description", limit=12)
    assert text == "person readi"
    assert len(text) == 12
    assert "\n" not in text
    assert "\u202e" not in text


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, "10", 2001])
def test_clean_text_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="text limit"):
        perception._clean_text("reading", field="camera description", limit=limit)


def test_clean_text_bounds_work_before_normalization(monkeypatch):
    monkeypatch.setattr(perception, "_MAX_TEXT_INPUT_CHARS", 20)
    text = perception._clean_text(
        "person reading " + "word " * 100_000,
        field="camera description",
        limit=1000,
    )
    assert text == "person reading word"


def test_wrap_text_preserves_normal_words_within_the_budget():
    assert perception._wrap_text("person reading a book", 10) == ["person", "reading a", "book"]


def test_wrap_text_splits_unbroken_model_tokens_to_the_budget():
    lines = perception._wrap_text("prefix abcdefghijkl suffix", 5)

    assert lines == ["prefi", "x", "abcde", "fghij", "kl", "suffi", "x"]
    assert all(len(line) <= 5 for line in lines)


@pytest.mark.parametrize("max_chars", [True, False, 0, -1, 1001, 1.5, "10"])
def test_wrap_text_rejects_invalid_budgets(max_chars):
    with pytest.raises(ValueError, match="text wrap limit"):
        perception._wrap_text("reading", max_chars)


def test_fit_label_lines_marks_text_clipped_to_the_frame_budget():
    assert perception._fit_label_lines(["one", "two", "three"], 2) == ["one", "tw…"]
    assert perception._fit_label_lines(["one", "two"], 2) == ["one", "two"]


@pytest.mark.parametrize("max_lines", [True, False, 0, -1, 1.5, "2"])
def test_fit_label_lines_rejects_invalid_budgets(max_lines):
    with pytest.raises(ValueError, match="label line budget"):
        perception._fit_label_lines(["reading"], max_lines)


def test_draw_label_keeps_background_and_baselines_inside_small_frame(monkeypatch):
    rectangles = []
    baselines = []
    monkeypatch.setattr(perception.cv2, "getTextSize", lambda *args: ((120, 10), 0))
    monkeypatch.setattr(
        perception.cv2, "rectangle",
        lambda _frame, start, end, *_args: rectangles.append((start, end)),
    )
    monkeypatch.setattr(perception.cv2, "addWeighted", lambda *_args: None)
    monkeypatch.setattr(
        perception.cv2, "putText",
        lambda _frame, _line, origin, *_args: baselines.append(origin),
    )
    frame = np.zeros((60, 80, 3), dtype=np.uint8)

    perception._draw_label(frame, "word " * 200, 75, 55, max_width=100)

    assert len(baselines) == 1
    assert rectangles
    for point in (*rectangles[0], *baselines):
        assert 0 <= point[0] < frame.shape[1]
        assert 0 <= point[1] < frame.shape[0]


def test_classifier_input_is_bounded_before_model_call(monkeypatch):
    seen = {}

    class Client:
        def generate(self, *, model, prompt):
            seen["prompt"] = prompt
            return type("Response", (), {"response": "no"})()

    monkeypatch.setattr(perception, "_ollama_client", Client())
    assert perception._qwen_health_check("working on homework", "x" * 10000) is True
    assert len(seen["prompt"]) < 4000


def test_keyword_shortcut_still_handles_clear_distraction_without_context(monkeypatch):
    class Client:
        def generate(self, **kwargs):
            raise AssertionError("clear context-free keywords should not need a model call")

    monkeypatch.setattr(perception, "_ollama_client", Client())
    assert perception._qwen_health_check("person scrolling social media") is False


def test_recent_context_can_override_keyword_shortcut(monkeypatch):
    seen = {}

    class Client:
        def generate(self, *, model, prompt):
            seen["prompt"] = prompt
            return type("Response", (), {"response": "no"})()

    monkeypatch.setattr(perception, "_ollama_client", Client())

    assert perception._qwen_health_check(
        "person watching television",
        "The user scheduled a documentary study session for this hour.",
    ) is True
    assert "documentary study session" in seen["prompt"]


def test_qwen_backend_error_is_redacted(monkeypatch):
    calls = []

    class Client:
        def generate(self, **kwargs):
            raise RuntimeError("api-key=super-secret")

    monkeypatch.setattr(perception, "_ollama_client", Client())
    monkeypatch.setattr(perception.console, "print", calls.append)
    assert perception._qwen_health_check("working on homework") is True
    rendered = " ".join(map(str, calls))
    assert "super-secret" not in rendered
    assert "RuntimeError" in rendered
