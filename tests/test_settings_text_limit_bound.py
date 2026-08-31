import pytest

from settings import env_secret, env_text


def test_text_parser_limits_are_bounded(monkeypatch):
    monkeypatch.delenv("WAKEUP_TEST_TEXT", raising=False)
    assert env_text("WAKEUP_TEST_TEXT", "ok", max_length=100_000) == "ok"
    with pytest.raises(ValueError, match="100000"):
        env_text("WAKEUP_TEST_TEXT", "ok", max_length=100_001)
    with pytest.raises(ValueError, match="100000"):
        env_secret("WAKEUP_TEST_TEXT", "", max_length=100_001)
