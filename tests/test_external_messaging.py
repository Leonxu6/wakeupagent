import os
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
        self.assertIn("不支持", result)
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
    def test_opted_in_message_redacts_body_and_resolved_contact_from_result_and_log(self, run, console_print):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True), \
             patch("config.WECHAT_CONTACTS", {"mentor": "Private Real Contact"}):
            result = send_wechat_shame_message.invoke({"target": "mentor", "message": "Please check in"})
        self.assertIn("发送消息", result)
        self.assertIn("mentor", result)
        self.assertNotIn("Please check in", result)
        self.assertNotIn("Private Real Contact", result)
        rendered_log = console_print.call_args.args[0]
        self.assertIn("mentor", rendered_log)
        self.assertNotIn("Private Real Contact", rendered_log)
        self.assertNotIn("Please check in", rendered_log)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:2], ["osascript", "-e"])

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
