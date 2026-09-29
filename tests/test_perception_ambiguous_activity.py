"""Ambiguous camera descriptions need context from the local classifier."""

import pytest

import perception


@pytest.mark.parametrize(
    "description",
    [
        "scrolling through lecture notes on a laptop",
        "watching a video lecture for class",
        "lying in bed while reading assigned material",
    ],
)
def test_ambiguous_activity_uses_local_classifier(monkeypatch, description):
    prompts = []

    class Client:
        def generate(self, *, model, prompt):
            prompts.append(prompt)
            return type("Response", (), {"response": "no"})()

    monkeypatch.setattr(perception, "_ollama_client", Client())
    assert perception._qwen_health_check(description) is True
    assert prompts and description in prompts[0]
    assert "lecture or tutorial" in prompts[0]
    assert "ambiguous, answer no" in prompts[0]


def test_explicit_social_media_still_uses_fast_path(monkeypatch):
    class Client:
        def generate(self, **kwargs):
            raise AssertionError("clear leisure activity should not call the model")

    monkeypatch.setattr(perception, "_ollama_client", Client())
    assert perception._qwen_health_check("scrolling social media") is False
