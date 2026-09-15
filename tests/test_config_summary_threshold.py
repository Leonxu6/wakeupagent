import pytest

import config


def test_summary_threshold_rejects_values_that_cannot_compress(monkeypatch):
    monkeypatch.setenv("WAKEUP_SUMMARIZE_THRESHOLD", "5")

    with pytest.raises(ValueError, match="at least 6"):
        config._summary_threshold()


def test_summary_threshold_accepts_first_compressible_value(monkeypatch):
    monkeypatch.setenv("WAKEUP_SUMMARIZE_THRESHOLD", "6")

    assert config._summary_threshold() == 6
