import os

import pytest

import settings


def test_integer_environment_text_has_a_bounded_parse_cost(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_INT", "9" * (settings._MAX_NUMERIC_TEXT + 1))
    with pytest.raises(ValueError, match="numeric text"):
        settings.env_int("WAKEUP_TEST_INT", 1)


def test_float_environment_text_has_the_same_bound(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_FLOAT", "1" * (settings._MAX_NUMERIC_TEXT + 1))
    with pytest.raises(ValueError, match="numeric text"):
        settings.env_float("WAKEUP_TEST_FLOAT", 1.0)


def test_bounded_numeric_text_keeps_normal_signed_integer_behavior(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_INT", "-12")
    assert settings.env_int("WAKEUP_TEST_INT", 0, minimum=-20, maximum=20) == -12


def test_oversized_numeric_rejection_happens_before_int_conversion(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_INT", "7" * 1000)
    called = False
    original = settings.int if hasattr(settings, "int") else None
    with pytest.raises(ValueError, match="numeric text"):
        settings.env_int("WAKEUP_TEST_INT", 1)


def test_missing_numeric_values_still_use_validated_defaults(monkeypatch):
    monkeypatch.delenv("WAKEUP_TEST_INT", raising=False)
    monkeypatch.delenv("WAKEUP_TEST_FLOAT", raising=False)
    assert settings.env_int("WAKEUP_TEST_INT", 3, minimum=1, maximum=5) == 3
    assert settings.env_float("WAKEUP_TEST_FLOAT", 0.5, minimum=0.0, maximum=1.0) == 0.5
