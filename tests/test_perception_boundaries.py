import numpy as np
import pytest

import perception


def test_latest_frame_returns_defensive_copy(monkeypatch):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", original)
    copy = perception.get_latest_frame()
    assert copy is not original
    copy[0, 0, 0] = 255
    assert original[0, 0, 0] == 0


def test_latest_frame_can_require_a_capture_after_request(monkeypatch):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(perception, "_latest_raw_frame", original)
    monkeypatch.setattr(perception, "_latest_frame_captured_at", 10.0)

    assert perception.get_latest_frame(captured_after=10.1) is None
    assert perception.get_latest_frame(captured_after=10.0) is not None


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
