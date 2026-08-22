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
    def test_valid_text_uses_bounded_say_invocation_without_echoing_prompt(self, run):
        run.return_value.returncode = 0
        prompt = "Back to the task."
        result = play_tts_punishment.invoke({"text": prompt})
        self.assertIn("TTS", result)
        self.assertNotIn(prompt, result)
        run.assert_called_once_with(
            ["say", "-v", "Tingting", prompt],
            timeout=60,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
