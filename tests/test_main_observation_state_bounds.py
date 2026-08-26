from main import _MESSAGE_TEXT_LIMIT, _observation_state


def test_observation_state_bounds_and_normalizes_text():
    state = _observation_state("x" * (_MESSAGE_TEXT_LIMIT + 20) + "\x00", "2026-08-26 17:00:00", True, False)
    assert len(state["current_vision_text"]) == _MESSAGE_TEXT_LIMIT
    assert "\x00" not in state["current_vision_text"]


def test_observation_state_normalizes_timestamp_whitespace():
    state = _observation_state("working", "2026-08-26\n17:00:00", True, False)
    assert state["timestamp"] == "2026-08-26 17:00:00"
