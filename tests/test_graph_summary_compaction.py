from unittest.mock import patch

from langchain_core.messages import HumanMessage, RemoveMessage

import graph


class _RecordingLLM:
    def __init__(self, *, content="updated summary", error=None):
        self.content = content
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return type("Response", (), {"content": self.content})()


def _messages(count):
    return [HumanMessage(content=f"event {index}", id=f"m{index}") for index in range(count)]


def test_summary_deletes_only_the_oldest_batch_it_actually_summarized():
    messages = _messages(30)
    llm = _RecordingLLM()

    with patch.object(graph, "_get_llm_plain", return_value=llm):
        result = graph._summarize_messages(messages, {"conversation_summary": "earlier"})

    assert [message.id for message in llm.messages[:-1]] == [f"m{i}" for i in range(20)]
    assert all(isinstance(message, RemoveMessage) for message in result["messages"])
    assert [message.id for message in result["messages"]] == [f"m{i}" for i in range(20)]
    assert result["conversation_summary"] == "updated summary"


def test_summary_failure_preserves_all_messages_for_a_later_retry():
    messages = _messages(30)
    llm = _RecordingLLM(error=RuntimeError("model unavailable"))

    with patch.object(graph, "_get_llm_plain", return_value=llm):
        result = graph._summarize_messages(messages, {"conversation_summary": "earlier"})

    assert result == {"conversation_summary": "earlier", "messages": []}


def test_empty_summary_response_preserves_all_messages_for_a_later_retry():
    messages = _messages(30)
    llm = _RecordingLLM(content=[])

    with patch.object(graph, "_get_llm_plain", return_value=llm):
        result = graph._summarize_messages(messages, {"conversation_summary": "earlier"})

    assert result == {"conversation_summary": "earlier", "messages": []}
