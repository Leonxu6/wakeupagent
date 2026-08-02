import unittest
from datetime import date
from unittest.mock import patch

from langchain_core.messages import HumanMessage, RemoveMessage

import graph
from main import _observation_state


class FixedDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 8, 1)


class DailyResetTests(unittest.TestCase):
    @patch.object(graph, "_save_daily_report")
    @patch.object(graph, "_generate_daily_report")
    def test_first_run_initializes_without_empty_report(self, generate_report, save_report):
        with patch.object(graph, "date", FixedDate):
            result = graph.daily_reset_node({"messages": []})

        self.assertEqual(result["session_date"], "2026-08-01")
        self.assertEqual(result["unhealthy_count"], 0)
        self.assertEqual(result["consecutive_healthy"], 0)
        self.assertEqual(result["react_iterations"], 0)
        generate_report.assert_not_called()
        save_report.assert_not_called()

    def test_same_day_is_a_noop(self):
        with patch.object(graph, "date", FixedDate):
            result = graph.daily_reset_node({"session_date": "2026-08-01"})

        self.assertEqual(result, {})

    @patch.object(graph, "_save_daily_report")
    @patch.object(graph, "_generate_daily_report", return_value="daily summary")
    def test_new_day_archives_previous_state(self, generate_report, save_report):
        message = HumanMessage(content="previous observation", id="message-1")
        state = {
            "messages": [message],
            "session_date": "2026-07-31",
            "unhealthy_count": 2,
            "consecutive_healthy": 3,
            "react_iterations": 4,
        }

        with patch.object(graph, "date", FixedDate):
            result = graph.daily_reset_node(state)

        generate_report.assert_called_once_with([message], "2026-07-31", state)
        save_report.assert_called_once_with("daily summary", "2026-07-31")
        self.assertEqual(result["session_date"], "2026-08-01")
        self.assertEqual(result["conversation_summary"], "daily summary")
        self.assertEqual(result["unhealthy_count"], 0)
        self.assertEqual(result["consecutive_healthy"], 0)
        self.assertEqual(result["react_iterations"], 0)
        self.assertEqual(len(result["messages"]), 1)
        self.assertIsInstance(result["messages"][0], RemoveMessage)
        self.assertEqual(result["messages"][0].id, "message-1")

    def test_observation_input_preserves_checkpointed_daily_state(self):
        result = _observation_state("reading", "12:00:00", True, False)

        self.assertNotIn("session_date", result)
        self.assertEqual(result["current_vision_text"], "reading")
        self.assertTrue(result["healthy"])
        self.assertFalse(result["should_escalate"])


if __name__ == "__main__":
    unittest.main()
