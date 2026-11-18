from types import SimpleNamespace
from unittest.mock import patch

import graph
from langchain_core.messages import AIMessage


class _FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, _messages):
        return SimpleNamespace(content=self.content)


class _FakeDecisionLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):
        return self.response


def test_daily_report_normalizes_structured_model_content():
    content = [{"type": "text", "text": "  focused\n\u202e tomorrow  "}, {"text": "plan"}]
    with patch.object(graph, "_get_llm_plain", return_value=_FakeLLM(content)):
        report = graph._generate_daily_report([], "2026-09-04", {"unhealthy_count": 2})

    assert report == "focused tomorrow plan"
    assert "\u202e" not in report


def test_daily_report_bounds_large_model_content():
    with patch.object(graph, "_get_llm_plain", return_value=_FakeLLM("x" * 10_000)):
        report = graph._generate_daily_report([], "2026-09-04", {"unhealthy_count": 1})

    assert len(report) == graph._REPORT_TEXT_LIMIT


def test_summary_normalizes_structured_model_content():
    content = [{"text": "  yesterday\nwork  "}, {"text": "\u202efocus"}]
    with patch.object(graph, "_get_llm_plain", return_value=_FakeLLM(content)):
        result = graph._summarize_messages([], {"conversation_summary": ""})

    assert result["conversation_summary"] == "yesterday work focus"


def test_summary_falls_back_to_normalized_existing_summary_for_empty_model_output():
    with patch.object(graph, "_get_llm_plain", return_value=_FakeLLM([])):
        result = graph._summarize_messages([], {"conversation_summary": "  old\n\u202esummary  "})

    assert result["conversation_summary"] == "old summary"


def test_decision_persists_only_normalized_model_text_and_validated_tool_calls():
    response = AIMessage(
        content="  act\n\u202e" + ("x" * 3_000),
        tool_calls=[
            {
                "name": "play_tts_punishment",
                "args": {},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
        additional_kwargs={"private_provider_payload": "discard me"},
        response_metadata={"backend_trace": "discard me"},
    )
    response.tool_calls[0]["provider_extension"] = "discard me"
    with patch.object(graph, "_get_llm", return_value=_FakeDecisionLLM(response)):
        result = graph.decision_node({"messages": []})

    persisted = result["messages"][-1]
    assert isinstance(persisted, AIMessage)
    assert persisted.content.startswith("act ")
    assert "\u202e" not in persisted.content
    assert len(persisted.content) == graph._RESPONSE_TEXT_LIMIT
    assert persisted.additional_kwargs == {}
    assert persisted.response_metadata == {}
    assert persisted.tool_calls == [
        {"name": "play_tts_punishment", "args": {}, "id": "call-1", "type": "tool_call"}
    ]
