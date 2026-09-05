import history
from history import ContextHistory


def test_render_budget_preserves_newest_whole_entry_with_summary(monkeypatch):
    monkeypatch.setattr(history, "_MAX_RENDER_CHARS", 50)
    context = ContextHistory(observation_limit=40)
    context.set_summary("focus")
    context.add_observation("first entry")
    context.add_observation("second entry")

    rendered = context.render(recent=2)

    assert rendered == "Summary: focus\n\nRecent history:\n[Obs] second entry"
    assert len(rendered) <= 50
