import pytest

from settings import env_float


def test_float_parser_accepts_canonical_ascii_forms(monkeypatch):
    for raw, expected in (("0", 0.0), ("-1.25", -1.25), (".5", 0.5), ("1.", 1.0), ("-1.25e2", -125.0)):
        monkeypatch.setenv("RATE", raw)
        assert env_float("RATE", 0.0) == expected


def test_float_parser_rejects_python_specific_or_non_ascii_forms(monkeypatch):
    for raw in ("+1", "1_000", "１２.５", "0x10"):
        monkeypatch.setenv("RATE", raw)
        with pytest.raises(ValueError, match="number"):
            env_float("RATE", 0.0)
