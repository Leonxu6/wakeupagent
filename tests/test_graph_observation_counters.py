import unittest

import graph


class ObservationCounterTests(unittest.TestCase):
    def test_escalated_observation_updates_daily_metrics_before_tool_selection(self):
        result = graph.perception_node(
            {
                "current_vision_text": "watching a movie",
                "timestamp": "2026-11-22 17:00:00",
                "should_escalate": True,
                "unhealthy_count": 2,
                "consecutive_healthy": 4,
            }
        )

        self.assertEqual(result["unhealthy_count"], 3)
        self.assertEqual(result["consecutive_healthy"], 0)

    def test_healthy_observation_advances_streak_without_fabricating_failure(self):
        result = graph.perception_node(
            {
                "current_vision_text": "reading project notes",
                "timestamp": "2026-11-22 17:01:00",
                "should_escalate": False,
                "unhealthy_count": 2,
                "consecutive_healthy": 4,
            }
        )

        self.assertNotIn("unhealthy_count", result)
        self.assertEqual(result["consecutive_healthy"], 5)

    def test_observation_metrics_recover_from_malformed_persisted_counters(self):
        result = graph.perception_node(
            {
                "current_vision_text": "working",
                "timestamp": "2026-11-22 17:02:00",
                "should_escalate": False,
                "consecutive_healthy": True,
            }
        )

        self.assertEqual(result["consecutive_healthy"], 1)


if __name__ == "__main__":
    unittest.main()
