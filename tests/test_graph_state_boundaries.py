import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

import graph


def test_state_counter_resets_malformed_persisted_values():
    assert graph._state_counter({"react_iterations": True}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": -1}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": "3"}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": graph._MAX_STATE_COUNTER + 1}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": 3}, "react_iterations") == 3


def test_state_messages_recovers_only_valid_bounded_messages():
    first = HumanMessage(content="hello")
    last = AIMessage(content="world")
    assert graph._state_messages({"messages": "corrupt"}) == []
    assert graph._state_messages({"messages": [first, None, "bad", last]}) == [first, last]

    many = [HumanMessage(content=str(index)) for index in range(graph._MAX_STATE_MESSAGES + 3)]
    recovered = graph._state_messages({"messages": many})
    assert len(recovered) == graph._MAX_STATE_MESSAGES
    assert recovered[0].content == "3"


def test_state_session_date_rejects_noncanonical_or_hostile_values():
    assert graph._state_session_date({"session_date": "2026-09-19"}) == "2026-09-19"
    for value in (None, 20260919, "2026-9-19", "2026-02-30", "2026-09-19\n"):
        assert graph._state_session_date({"session_date": value}) == ""


def test_routes_fail_closed_on_malformed_persisted_values():
    assert graph.route_after_perception({"should_escalate": "false"}) == END
    assert graph.route_after_perception({"should_escalate": True}) == "decision"
    assert graph.route_after_decision({"messages": "corrupt"}) == END


def test_perception_normalizes_persisted_display_fields():
    result = graph.perception_node(
        {
            "current_vision_text": "working\nquietly\u200d",
            "timestamp": "2026-09-19\n17:00",
            "should_escalate": False,
        }
    )
    content = result["messages"][0].content
    assert "working quietly" in content
    assert "\u200d" not in content
    assert "2026-09-19 17:00" in content


def test_tool_call_names_reject_malformed_envelopes():
    with pytest.raises(ValueError):
        graph._tool_call_names("not-a-list")
    with pytest.raises(ValueError):
        graph._tool_call_names([None])
    with pytest.raises(ValueError):
        graph._tool_call_names([{"name": " bad"}])
    with pytest.raises(ValueError):
        graph._tool_call_names([{"name": "x"}] * (graph._MAX_TOOL_CALLS + 1))


def test_tool_call_names_preserve_valid_order():
    assert graph._tool_call_names([{"name": "observe_camera"}, {"name": "open_webpage"}]) == [
        "observe_camera",
        "open_webpage",
    ]


def test_repair_skips_malformed_persisted_tool_calls_without_crashing():
    message = AIMessage(content="hello")
    message.tool_calls = [None, {"id": 123, "name": "observe_camera"}]

    ordered, repairs = graph._reorder_and_repair([message])

    assert ordered == [message]
    assert repairs == []
