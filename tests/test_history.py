import unittest

from history import ContextHistory


class ContextHistoryTests(unittest.TestCase):
    def test_buffer_drops_oldest_items_at_capacity(self):
        history = ContextHistory(max_items=2)
        history.add_observation("first")
        history.add_decision("second")
        history.add_observation("third")
        self.assertEqual(len(history), 2)
        rendered = history.render(recent=10)
        self.assertNotIn("first", rendered)
        self.assertIn("second", rendered)
        self.assertIn("third", rendered)

    def test_summary_and_entries_are_trimmed_and_bounded(self):
        history = ContextHistory(max_items=3, summary_limit=4, observation_limit=5, decision_limit=6)
        history.set_summary("  summary  ")
        history.add_observation("  observation  ")
        history.add_decision("  decision  ")
        rendered = history.render()
        self.assertIn("Summary: summ", rendered)
        self.assertIn("[Obs] obser", rendered)
        self.assertIn("[Brain] decisi", rendered)

    def test_non_text_items_are_ignored(self):
        history = ContextHistory()
        history.set_summary(None)
        history.add_observation(None)
        history.add_decision(7)
        self.assertEqual(history.render(), "")

    def test_invalid_limits_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContextHistory(max_items=value)
        history = ContextHistory()
        for value in (0, -2, True, "2"):
            with self.subTest(recent=value), self.assertRaises(ValueError):
                history.render(recent=value)


if __name__ == "__main__":
    unittest.main()
