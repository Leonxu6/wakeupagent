import unittest

import diagnostics


class DiagnosticPersistenceBidiTests(unittest.TestCase):
    def test_persistence_paths_reject_directional_controls(self):
        for value in ("memory\u202e/state.db", "memory/report\u2066.md"):
            with self.subTest(value=value):
                check = diagnostics._persistence_parent_check("state", value)
                self.assertFalse(check.ok)
                self.assertIn("controls", check.detail)


if __name__ == "__main__":
    unittest.main()
