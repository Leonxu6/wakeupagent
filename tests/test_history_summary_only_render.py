import pytest

from history import ContextHistory


def test_render_zero_recent_items_returns_summary_only():
    history = ContextHistory()
    history.set_summary("standing desk session")
    history.add_observation("looking at editor")

    assert history.render(recent=0) == "Summary: standing desk session"


def test_render_rejects_negative_or_boolean_recent_counts():
    history = ContextHistory()

    for value in (-1, True):
        with pytest.raises(ValueError):
            history.render(recent=value)
