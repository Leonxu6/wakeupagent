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


@pytest.mark.parametrize("frame", [None, object(), np.array([]), np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 5))])
def test_validate_frame_rejects_invalid_shapes(frame):
    with pytest.raises(ValueError):
        perception._validate_frame(frame)


def test_clean_text_normalizes_controls_and_bounds_output():
    text = perception._clean_text("  person\nreading\u202etext  ", field="camera description", limit=12)
    assert text == "person readin"
    assert "\n" not in text
    assert "\u202e" not in text


def test_classifier_input_is_bounded_before_model_call(monkeypatch):
    seen = {}

    class Client:
        def generate(self, *, model, prompt):
            seen["prompt"] = prompt
            return type("Response", (), {"response": "no"})()

    monkeypatch.setattr(perception, "_ollama_client", Client())
    assert perception._qwen_health_check("working on homework", "x" * 10000) is True
    assert len(seen["prompt"]) < 4000


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
