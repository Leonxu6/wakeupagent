import pytest
from langchain_core.messages import AIMessage

import graph


def test_state_counter_resets_malformed_persisted_values():
    assert graph._state_counter({"react_iterations": True}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": -1}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": "3"}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": graph._MAX_STATE_COUNTER + 1}, "react_iterations") == 0
    assert graph._state_counter({"react_iterations": 3}, "react_iterations") == 3


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
