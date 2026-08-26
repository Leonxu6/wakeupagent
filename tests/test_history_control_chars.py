from history import ContextHistory


def test_history_replaces_control_characters_with_safe_spacing():
    history = ContextHistory(observation_limit=100)
    history.add_observation("working\x00on\x7ftask")
    assert "[Obs] working on task" in history.render()
    assert "\x00" not in history.render()
    assert "\x7f" not in history.render()
