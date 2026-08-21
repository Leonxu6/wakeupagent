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
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True):
            result = send_wechat_shame_message.invoke({"target": "unknown", "message": "Status update"})
        self.assertIn("不支持", result)
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_invalid_message_is_rejected_before_automation(self, run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True):
            for message in ("", " padded", "bad\x00message", "x" * 501):
                with self.subTest(message=message[:20]):
                    self.assertIn("Error", send_wechat_shame_message.invoke({"target": "导师", "message": message}))
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_opted_in_message_uses_osascript_and_avoids_echoing_body_in_result(self, run):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        with patch.dict(os.environ, {"WAKEUP_ALLOW_EXTERNAL_MESSAGING": "true"}, clear=True):
            result = send_wechat_shame_message.invoke({"target": "导师", "message": "Please check in"})
        self.assertIn("发送消息", result)
        self.assertNotIn("Please check in", result)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:2], ["osascript", "-e"])


if __name__ == "__main__":
    unittest.main()
