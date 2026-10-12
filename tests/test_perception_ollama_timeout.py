import pytest

import config
import perception


def test_ollama_timeout_defaults_to_one_minute(monkeypatch):
    monkeypatch.delenv("WAKEUP_OLLAMA_TIMEOUT_SEC", raising=False)
    assert config._ollama_timeout() == 60.0


@pytest.mark.parametrize("value", ["0.9", "601", "nan", "inf", "true"])
def test_ollama_timeout_rejects_invalid_or_unbounded_values(monkeypatch, value):
    monkeypatch.setenv("WAKEUP_OLLAMA_TIMEOUT_SEC", value)
    with pytest.raises(ValueError):
        config._ollama_timeout()


def test_ollama_timeout_accepts_documented_boundaries(monkeypatch):
    monkeypatch.setenv("WAKEUP_OLLAMA_TIMEOUT_SEC", "1")
    assert config._ollama_timeout() == 1.0
    monkeypatch.setenv("WAKEUP_OLLAMA_TIMEOUT_SEC", "600")
    assert config._ollama_timeout() == 600.0


def test_perception_client_receives_explicit_http_timeout(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(perception.ollama, "Client", Client)
    client = perception._create_ollama_client("http://localhost:11434", 42.5)

    assert isinstance(client, Client)
    assert captured == {"host": "http://localhost:11434", "timeout": 42.5}
