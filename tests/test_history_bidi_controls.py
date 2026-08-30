import unittest

from history import ContextHistory


class HistoryBidiControlTests(unittest.TestCase):
    def test_context_entries_neutralize_directional_controls(self):
        history = ContextHistory(max_items=4, observation_limit=40, decision_limit=40)
        history.add_observation("reading\u202eexe")
        history.add_decision("continue\u2066focus")
        rendered = history.render()
        self.assertNotIn("\u202e", rendered)
        self.assertNotIn("\u2066", rendered)
        self.assertIn("[Obs] reading exe", rendered)
        self.assertIn("[Brain] continue focus", rendered)

    def test_snapshot_restore_rejects_directional_controls(self):
        snapshot = {
            "version": 1,
            "limits": {"max_items": 2, "summary_limit": 40, "observation_limit": 40, "decision_limit": 40},
            "summary": "safe\u202eevil",
            "items": [],
        }
        with self.assertRaisesRegex(ValueError, "summary"):
            ContextHistory.from_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
