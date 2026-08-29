from history import ContextHistory


def test_truncation_does_not_leave_snapshot_payload_with_trailing_space():
    history = ContextHistory(max_items=2, observation_limit=4, decision_limit=4, summary_limit=4)
    history.add_observation("ab cd")
    history.add_decision("xy zw")
    history.set_summary("hi jk")

    snapshot = history.snapshot()

    assert snapshot["summary"] == "hi"
    assert snapshot["items"] == ["[Obs] ab", "[Brain] xy"]
    restored = ContextHistory.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
