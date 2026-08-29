import copy

import pytest

from history import ContextHistory


def test_history_snapshot_round_trips_without_sharing_mutable_state():
    history = ContextHistory(max_items=3, summary_limit=20, observation_limit=20, decision_limit=20)
    history.set_summary("today")
    history.add_observation("desk")
    history.add_decision("focus")
    snapshot = history.snapshot()
    restored = ContextHistory.from_snapshot(snapshot)
    assert restored.render() == history.render()
    snapshot["items"].append("[Obs] mutated")
    assert "mutated" not in restored.render()


def test_snapshot_is_json_friendly_and_explicitly_versioned():
    history = ContextHistory(max_items=2)
    snapshot = history.snapshot()
    assert snapshot["version"] == 1
    assert snapshot["limits"]["max_items"] == 2
    assert snapshot["items"] == []


def test_restore_rejects_unknown_versions_and_limit_shapes():
    snapshot = ContextHistory().snapshot()
    for bad in (None, [], {"version": 2}, {"version": 1, "limits": []}):
        with pytest.raises(ValueError):
            ContextHistory.from_snapshot(bad)
    bad = copy.deepcopy(snapshot)
    bad["limits"]["extra"] = 1
    with pytest.raises(ValueError):
        ContextHistory.from_snapshot(bad)


def test_restore_rejects_unknown_or_missing_top_level_fields():
    snapshot = ContextHistory().snapshot()
    extra = copy.deepcopy(snapshot)
    extra["unexpected"] = True
    missing = copy.deepcopy(snapshot)
    del missing["summary"]
    for bad in (extra, missing):
        with pytest.raises(ValueError, match="unknown fields|incomplete"):
            ContextHistory.from_snapshot(bad)


def test_restore_rejects_untrusted_or_unbounded_entries():
    snapshot = ContextHistory(max_items=2, observation_limit=5).snapshot()
    cases = []
    too_many = copy.deepcopy(snapshot)
    too_many["items"] = ["[Obs] one", "[Obs] two", "[Obs] tri"]
    cases.append(too_many)
    bad_prefix = copy.deepcopy(snapshot)
    bad_prefix["items"] = ["[Tool] hidden"]
    cases.append(bad_prefix)
    oversized = copy.deepcopy(snapshot)
    oversized["items"] = ["[Obs] abcdef"]
    cases.append(oversized)
    control = copy.deepcopy(snapshot)
    control["items"] = ["[Obs] a\tb"]
    cases.append(control)
    duplicates = copy.deepcopy(snapshot)
    duplicates["items"] = ["[Obs] one", "[Obs] one"]
    cases.append(duplicates)
    for bad in cases:
        with pytest.raises(ValueError):
            ContextHistory.from_snapshot(bad)


def test_restore_rejects_non_normalized_summary():
    snapshot = ContextHistory(summary_limit=10).snapshot()
    snapshot["summary"] = "  padded  "
    with pytest.raises(ValueError):
        ContextHistory.from_snapshot(snapshot)
