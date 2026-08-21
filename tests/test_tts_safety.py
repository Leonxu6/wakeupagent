import unittest
from unittest.mock import patch

from tools import play_tts_punishment


class TtsSafetyTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    def test_rejects_empty_padded_control_and_multiline_text(self, run):
        invalid = ("", " padded", "padded ", "bad\x00text", "line one\nline two")
        for text in invalid:
            with self.subTest(text=text):
                self.assertIn("Error", play_tts_punishment.invoke({"text": text}))
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_rejects_oversize_text(self, run):
        result = play_tts_punishment.invoke({"text": "x" * 201})
        self.assertIn("at most 200", result)
        run.assert_not_called()

    @patch("tools.subprocess.run")
    def test_valid_text_uses_bounded_say_invocation(self, run):
        run.return_value.returncode = 0
        result = play_tts_punishment.invoke({"text": "Back to the task."})
        self.assertIn("TTS", result)
        run.assert_called_once_with(
            ["say", "-v", "Tingting", "Back to the task."],
            timeout=60,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
