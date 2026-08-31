from history import ContextHistory


def test_render_limit_never_cuts_a_history_entry_mid_line():
    history = ContextHistory(max_items=6, observation_limit=10_000)
    for prefix in "abcdef":
        history.add_observation(prefix + "x" * 9_999)

    rendered = history.render(recent=6)
    lines = [line for line in rendered.splitlines() if line.startswith("[Obs] ")]
    assert 1 <= len(lines) < 6
    assert all(len(line) == len("[Obs] ") + 10_000 for line in lines)
    assert len(rendered) <= 50_000
