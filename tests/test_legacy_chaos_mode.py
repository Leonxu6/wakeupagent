import unittest
from unittest.mock import patch

import tools


class LegacyChaosModeTests(unittest.TestCase):
    @patch("tools.subprocess.run")
    @patch("tools.subprocess.Popen")
    def test_legacy_entry_point_has_no_side_effects(self, popen, run):
        result = tools.chaos_terminal_punishment.invoke({"message": "anything"})
        self.assertIn("disabled", result)
        run.assert_not_called()
        popen.assert_not_called()

    def test_legacy_entry_point_is_not_exposed_to_the_agent(self):
        names = {tool.name for tool in tools.ALL_TOOLS}
        self.assertNotIn("chaos_terminal_punishment", names)
        self.assertEqual(
            names,
            {
                "play_tts_punishment",
                "send_wechat_shame_message",
                "open_webpage",
                "force_close_app",
                "observe_camera",
            },
        )


if __name__ == "__main__":
    unittest.main()
