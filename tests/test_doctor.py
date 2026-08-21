import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import doctor


class DoctorTests(unittest.TestCase):
    def test_reports_missing_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = doctor.collect_diagnostics(Path(tmp))
        models = [result for result in results if result.name.startswith("model:")]
        self.assertEqual(len(models), 2)
        self.assertTrue(all(not result.ok for result in models))

    def test_reports_present_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename in doctor._MODEL_FILES:
                (root / filename).write_bytes(b"model")
            results = doctor.collect_diagnostics(root)
        models = [result for result in results if result.name.startswith("model:")]
        self.assertTrue(all(result.ok for result in models))

    def test_disruptive_action_summary_lists_enabled_flags(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(doctor.config, "ENABLE_WECHAT_ACTIONS", True),
            patch.object(doctor.config, "ENABLE_APP_TERMINATION", False),
            patch.object(doctor.config, "ENABLE_CHAOS_ACTIONS", True),
        ):
            results = doctor.collect_diagnostics(Path(tmp))
        check = next(result for result in results if result.name == "disruptive_actions")
        self.assertFalse(check.ok)
        self.assertIn("wechat", check.detail)
        self.assertIn("chaos", check.detail)

    def test_optional_failures_do_not_fail_required_summary(self):
        results = [
            doctor.CheckResult("required", True, "ok"),
            doctor.CheckResult("optional", False, "warning", required=False),
        ]
        self.assertTrue(doctor.required_checks_pass(results))

    def test_required_failure_fails_summary(self):
        results = [doctor.CheckResult("required", False, "bad")]
        self.assertFalse(doctor.required_checks_pass(results))


if __name__ == "__main__":
    unittest.main()
