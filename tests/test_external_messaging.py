import os
import subprocess
import unittest
from unittest.mock import patch

from tools import send_wechat_shame_message


class ExternalMessagingTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    def test_messaging_is_disabled_by_default(self, run):
        with patch.dict(os.environ, {}, clear=True):
            result = send_wechat_shame_message.invoke({"target": "导师", "message": "Status update"})
        self.assertIn("disabled", result)
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_invalid_target_alias_does_not_start_automation(self, run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True), \
             patch("config.WECHAT_CONTACTS", {"mentor": "Private Real Contact", "family": "Another Contact"}):
            result = send_wechat_shame_message.invoke({"target": "unknown", "message": "Status update"})
        self.assertIn("unsupported", result)
        self.assertNotIn("mentor", result)
        self.assertNotIn("family", result)
        self.assertNotIn("Private Real Contact", result)
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_invalid_message_is_rejected_before_automation(self, run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True):
            for message in ("", " padded", "bad\x00message", "x" * 501):
                with self.subTest(message=message[:20]):
                    self.assertIn("Error", send_wechat_shame_message.invoke({"target": "导师", "message": message}))
        run.assert_not_called()

    @patch("tools.console.print")
    @patch("tools.subprocess.run")
    def test_opted_in_message_redacts_body_alias_and_resolved_contact(self, run, console_print):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True), \
             patch("config.WECHAT_CONTACTS", {"mentor": "Private Real Contact"}):
            result = send_wechat_shame_message.invoke({"target": "mentor", "message": "Please check in"})
        self.assertEqual(result, "消息发送完成")
        self.assertNotIn("mentor", result)
        self.assertNotIn("Please check in", result)
        self.assertNotIn("Private Real Contact", result)
        rendered_log = console_print.call_args.args[0]
        self.assertNotIn("mentor", rendered_log)
        self.assertNotIn("Private Real Contact", rendered_log)
        self.assertNotIn("Please check in", rendered_log)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["osascript"])
        self.assertIn("Private Real Contact", run.call_args.kwargs["input"])
        self.assertIn("Please check in", run.call_args.kwargs["input"])

    @patch("tools.subprocess.run")
    def test_private_message_is_supplied_over_stdin_not_process_arguments(self, run):
        run.return_value = subprocess.CompletedProcess(["osascript"], 0)
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True), \
             patch("config.WECHAT_CONTACTS", {"mentor": "Private Real Contact"}):
            result = send_wechat_shame_message.invoke({"target": "mentor", "message": "Private update"})

        self.assertEqual(result, "消息发送完成")
        argv = run.call_args.args[0]
        self.assertEqual(argv, ["osascript"])
        self.assertNotIn("Private Real Contact", " ".join(argv))
        self.assertNotIn("Private update", " ".join(argv))
        self.assertIn("Private Real Contact", run.call_args.kwargs["input"])
        self.assertIn("Private update", run.call_args.kwargs["input"])

    @patch("tools.subprocess.run")
    def test_automation_failures_do_not_echo_private_script_details(self, run):
        run.return_value.returncode = 1
        run.return_value.stderr = "failed around Private Real Contact and Please check in"
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True), \
             patch("config.WECHAT_CONTACTS", {"mentor": "Private Real Contact"}):
            result = send_wechat_shame_message.invoke({"target": "mentor", "message": "Please check in"})
        self.assertEqual(result, "Error: 微信自动化执行失败")
        self.assertNotIn("Private Real Contact", result)
        self.assertNotIn("Please check in", result)


if __name__ == "__main__":
    unittest.main()
