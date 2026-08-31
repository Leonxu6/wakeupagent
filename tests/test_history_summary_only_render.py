import pytest

from history import ContextHistory


def test_render_summary_returns_summary_without_recent_entries():
    history = ContextHistory()
    history.set_summary("standing desk session")
    history.add_observation("looking at editor")

    assert history.render_summary() == "Summary: standing desk session"


def test_render_still_rejects_zero_negative_and_boolean_recent_counts():
    history = ContextHistory()

    for value in (0, -1, True):
        with pytest.raises(ValueError):
            history.render(recent=value)
