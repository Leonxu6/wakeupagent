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
