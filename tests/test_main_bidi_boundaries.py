import unittest

import main


class MainBidiBoundaryTests(unittest.TestCase):
    def test_observation_state_neutralizes_directional_controls(self):
        state = main._observation_state("reading\u202ebook", "2026-08-30\u2066 17:00", True, False)
        self.assertEqual(state["current_vision_text"], "reading book")
        self.assertEqual(state["timestamp"], "2026-08-30 17:00")

    def test_message_text_neutralizes_directional_controls(self):
        self.assertEqual(main._message_text("left\u200fright"), "left right")
        self.assertEqual(main._message_text([{"text": "safe\u2066text"}]), "safe text")

    def test_error_logs_neutralize_directional_controls(self):
        self.assertEqual(main._log_error(RuntimeError("before\u202eafter")), "before after")


if __name__ == "__main__":
    unittest.main()
