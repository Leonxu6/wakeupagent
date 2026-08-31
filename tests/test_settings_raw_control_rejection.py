import pytest

from settings import env_bool, env_float, env_int


def test_raw_parsers_reject_bidirectional_controls(monkeypatch):
    cases = (
        ("COUNT", "1\u202e", lambda: env_int("COUNT", 0)),
        ("RATE", "1\u202e.5", lambda: env_float("RATE", 0.0)),
        ("FLAG", "tr\u202eue", lambda: env_bool("FLAG", False)),
    )
    for name, value, parser in cases:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match="control"):
            parser()
