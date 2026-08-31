import pytest

from history import ContextHistory


def test_history_snapshots_refuse_excessive_total_text():
    history = ContextHistory(max_items=101, observation_limit=10_000)
    for index in range(101):
        history.add_observation(f"{index:03d}" + "x" * 9_997)

    with pytest.raises(ValueError, match="1000000"):
        history.snapshot()
