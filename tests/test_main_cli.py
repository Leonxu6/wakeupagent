import unittest
from unittest.mock import patch

import doctor
import main


class MainCliTests(unittest.TestCase):
    def test_observation_state_does_not_overwrite_durable_counters(self):
        state = main._observation_state("working", "12:00:00", True, False)
        self.assertEqual(
            state,
            {
                "current_vision_text": "working",
                "healthy": True,
                "should_escalate": False,
                "timestamp": "12:00:00",
            },
        )
        self.assertNotIn("session_date", state)
        self.assertNotIn("unhealthy_count", state)

    def test_build_context_keeps_configured_number_of_recent_items(self):
        items = [f"item-{index}" for index in range(20)]
        context = main._build_context("summary", items, window=15)

        self.assertIn("Summary: summary", context)
        self.assertNotIn("item-4\n", context)
        self.assertIn("item-5", context)
        self.assertIn("item-19", context)
        self.assertEqual(context.count("item-"), 15)

    def test_build_context_handles_empty_summary_and_history(self):
        self.assertEqual(main._build_context("", []), "")
        self.assertEqual(main._build_context("summary", []), "Summary: summary")

    def test_build_context_truncates_long_summary(self):
        context = main._build_context("x" * 250, [])
        self.assertEqual(context, "Summary: " + "x" * 200)

    @patch("doctor.required_checks_pass", return_value=True)
    @patch("doctor.collect_diagnostics")
    def test_doctor_returns_zero_when_required_checks_pass(self, collect, required):
        collect.return_value = [doctor.CheckResult("python", True, "3.12")]
        self.assertEqual(main.run_doctor_mode(), 0)
        required.assert_called_once_with(collect.return_value)

    @patch("doctor.required_checks_pass", return_value=False)
    @patch("doctor.collect_diagnostics")
    def test_doctor_returns_nonzero_on_required_failure(self, collect, required):
        collect.return_value = [doctor.CheckResult("model", False, "missing")]
        self.assertEqual(main.run_doctor_mode(), 1)
        required.assert_called_once_with(collect.return_value)


if __name__ == "__main__":
    unittest.main()
