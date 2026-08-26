import pytest

from main import _observation_state


def test_observation_state_rejects_truthy_non_boolean_health_flag():
    with pytest.raises(ValueError, match="health flags must be boolean"):
        _observation_state("working", "2026-08-26 17:00:00", 1, False)


def test_observation_state_rejects_truthy_non_boolean_escalation_flag():
    with pytest.raises(ValueError, match="health flags must be boolean"):
        _observation_state("working", "2026-08-26 17:00:00", True, "yes")
