import unittest

from history import ContextHistory, _bounded_text


class HistoryLimitCapTests(unittest.TestCase):
    def test_history_constructor_caps_external_limits(self):
        for kwargs in (
            {"max_items": 501},
            {"summary_limit": 10_001},
            {"observation_limit": 10_001},
            {"decision_limit": 10_001},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "at most"):
                ContextHistory(**kwargs)

    def test_render_caps_requested_recent_window(self):
        history = ContextHistory()
        with self.assertRaisesRegex(ValueError, "at most 500"):
            history.render(recent=501)

    def test_bounded_text_rejects_unbounded_limits(self):
        with self.assertRaisesRegex(ValueError, "at most 10000"):
            _bounded_text("safe", limit=10_001)


if __name__ == "__main__":
    unittest.main()
