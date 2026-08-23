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

    def test_multiline_and_repeated_whitespace_are_normalized(self):
        history = ContextHistory()
        history.set_summary("first\nsecond")
        history.add_observation("look\taway\nnow")
        history.add_decision("  resume   work  ")
        self.assertEqual(
            history.render(),
            "Summary: first second\n\nRecent history:\n[Obs] look away now\n[Brain] resume work",
        )

    def test_non_text_items_are_ignored(self):
        history = ContextHistory()
        history.set_summary(None)
        history.add_observation(None)
        history.add_decision(7)
        self.assertEqual(history.render(), "")

    def test_malformed_summary_updates_do_not_erase_existing_context(self):
        history = ContextHistory()
        history.set_summary("keep this summary")
        for value in (None, 7, "", "   "):
            with self.subTest(value=value):
                history.set_summary(value)
                self.assertEqual(history.render(), "Summary: keep this summary")

    def test_invalid_limits_are_rejected(self):
        fields = ("max_items", "summary_limit", "observation_limit", "decision_limit")
        for field in fields:
            for value in (0, -1, True, 1.5, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ContextHistory(**{field: value})

        history = ContextHistory()
        for value in (0, -2, True, "2"):
            with self.subTest(recent=value), self.assertRaises(ValueError):
                history.render(recent=value)

    def test_limit_validation_preserves_valid_custom_sizes(self):
        history = ContextHistory(max_items=2, summary_limit=3, observation_limit=4, decision_limit=5)
        history.set_summary("abcdef")
        history.add_observation("abcdef")
        history.add_decision("abcdef")
        self.assertEqual(history.render(), "Summary: abc\n\nRecent history:\n[Obs] abcd\n[Brain] abcde")


if __name__ == "__main__":
    unittest.main()
