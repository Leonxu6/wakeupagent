import os
import unittest
from unittest.mock import patch

from tools import force_close_app


class ProcessControlTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    def test_process_control_is_disabled_by_default(self, run):
        with patch.dict(os.environ, {}, clear=True):
            result = force_close_app.invoke({"app_name": "Safari"})
        self.assertIn("disabled", result)
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_rejects_injection_like_app_names_before_subprocess(self, run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_PROCESS_CONTROL": "true"}, clear=True):
            for name in ('Safari" to quit', "Steam; rm -rf ~", "$(whoami)", "../Safari"):
                with self.subTest(name=name):
                    self.assertIn("Error", force_close_app.invoke({"app_name": name}))
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_enabled_process_control_uses_exact_app_name(self, run):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        with patch.dict(os.environ, {"WAKEUP_ALLOW_PROCESS_CONTROL": "true"}, clear=True):
            result = force_close_app.invoke({"app_name": "Safari"})
        self.assertIn("osascript", result)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["osascript", "-e"])
        self.assertIn('tell application "Safari" to quit', command[2])

    @patch("tools.subprocess.run")
    def test_no_fuzzy_pkill_fallback_is_used(self, run):
        first = unittest.mock.Mock(returncode=1, stderr="not running")
        second = unittest.mock.Mock(returncode=1, stderr="not found")
        run.side_effect = [first, second]
        with patch.dict(os.environ, {"WAKEUP_ALLOW_PROCESS_CONTROL": "true"}, clear=True):
            result = force_close_app.invoke({"app_name": "Safari"})
        self.assertIn("Error", result)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[1].args[0], ["killall", "Safari"])


if __name__ == "__main__":
    unittest.main()
