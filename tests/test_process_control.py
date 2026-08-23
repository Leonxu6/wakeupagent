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
        self.assertIn("正常退出", result)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["osascript", "-e"])
        self.assertIn('tell application "Safari" to quit', command[2])

    @patch("tools.subprocess.run")
    def test_failed_graceful_quit_never_falls_back_to_killall(self, run):
        run.return_value.returncode = 1
        run.return_value.stderr = "not running"
        with patch.dict(os.environ, {"WAKEUP_ALLOW_PROCESS_CONTROL": "true"}, clear=True):
            result = force_close_app.invoke({"app_name": "Safari"})
        self.assertIn("Error", result)
        self.assertIn("not running", result)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:2], ["osascript", "-e"])


if __name__ == "__main__":
    unittest.main()
