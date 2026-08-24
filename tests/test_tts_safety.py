import os
import subprocess
import unittest
from unittest.mock import patch

from tools import play_tts_punishment


class TtsSafetyTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    def test_tts_is_disabled_until_explicitly_enabled(self, run):
        with patch.dict(os.environ, {}, clear=True):
            result = play_tts_punishment.invoke({"text": "Back to the task."})
        self.assertIn("disabled", result)
        self.assertIn("WAKEUP_ALLOW_TTS", result)
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_rejects_empty_padded_control_and_multiline_text(self, run):
        invalid = ("", " padded", "padded ", "bad\x00text", "line one\nline two")
        with patch.dict(os.environ, {"WAKEUP_ALLOW_TTS": "true"}, clear=True):
            for text in invalid:
                with self.subTest(text=text):
                    self.assertIn("Error", play_tts_punishment.invoke({"text": text}))
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_rejects_oversize_text(self, run):
        with patch.dict(os.environ, {"WAKEUP_ALLOW_TTS": "true"}, clear=True):
            result = play_tts_punishment.invoke({"text": "x" * 201})
        self.assertIn("at most 200", result)
        run.assert_not_called()

    @patch("tools.console.print")
    @patch("tools.subprocess.run")
    def test_valid_text_uses_bounded_say_invocation_without_echoing_prompt(self, run, console_print):
        run.return_value.returncode = 0
        prompt = "Private reminder: review the draft."
        with patch.dict(os.environ, {"WAKEUP_ALLOW_TTS": "true"}, clear=True):
            result = play_tts_punishment.invoke({"text": prompt})
        self.assertIn("TTS", result)
        self.assertNotIn(prompt, result)
        rendered_log = console_print.call_args.args[0]
        self.assertIn(str(len(prompt)), rendered_log)
        self.assertNotIn(prompt, rendered_log)
        run.assert_called_once_with(
            ["say", "-v", "Tingting", prompt],
            timeout=60,
            check=True,
        )

    @patch("tools.subprocess.run")
    def test_fallback_failures_do_not_echo_spoken_text(self, run):
        prompt = "Private reminder with account details"
        run.side_effect = [
            subprocess.CalledProcessError(1, ["say", prompt]),
            RuntimeError(f"failed while speaking {prompt}"),
        ]
        with patch.dict(os.environ, {"WAKEUP_ALLOW_TTS": "true"}, clear=True):
            result = play_tts_punishment.invoke({"text": prompt})
        self.assertEqual(result, "Error: TTS 播放失败")
        self.assertNotIn(prompt, result)


if __name__ == "__main__":
    unittest.main()
