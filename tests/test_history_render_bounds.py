from history import ContextHistory, _MAX_RENDER_CHARS


def test_rendered_context_has_a_hard_total_size_cap():
    history = ContextHistory(
        max_items=500,
        summary_limit=10_000,
        observation_limit=10_000,
        decision_limit=10_000,
    )
    history.set_summary("s" * 10_000)
    for index in range(20):
        history.add_observation(f"{index:02d} " + "x" * 9_990)
    rendered = history.render(recent=20)
    assert len(rendered) <= _MAX_RENDER_CHARS
    assert rendered.startswith("Summary: ")
    assert not rendered.endswith(" ")


def test_default_rendering_remains_unchanged_below_cap():
    history = ContextHistory()
    history.set_summary("keep focus")
    history.add_observation("working")
    assert history.render() == "Summary: keep focus\n\nRecent history:\n[Obs] working"
