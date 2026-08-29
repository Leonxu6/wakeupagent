from history import ContextHistory


def test_truncation_does_not_leave_snapshot_payload_with_trailing_space():
    history = ContextHistory(max_items=2, observation_limit=4, decision_limit=4, summary_limit=4)
    history.add_observation("abc def")
    history.add_decision("xyz uvw")
    history.set_summary("hij klm")

    snapshot = history.snapshot()

    assert snapshot["summary"] == "hij"
    assert snapshot["items"] == ["[Obs] abc", "[Brain] xyz"]
    restored = ContextHistory.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
