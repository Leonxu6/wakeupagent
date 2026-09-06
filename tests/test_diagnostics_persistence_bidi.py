import unittest

import diagnostics


class DiagnosticPersistenceBidiTests(unittest.TestCase):
    def test_persistence_paths_reject_directional_controls(self):
        for value in ("memory\u202e/state.db", "memory/report\u2066.md"):
            with self.subTest(value=value):
                check = diagnostics._persistence_parent_check("state", value)
                self.assertFalse(check.ok)
                self.assertIn("controls", check.detail)

    def test_persistence_paths_reject_generic_format_and_surrogate_controls(self):
        for value in ("memory\u200c/state.db", "memory/report\u206a.md", "memory/" + chr(0xD800) + "state.db"):
            with self.subTest(value=value):
                check = diagnostics._persistence_parent_check("state", value)
                self.assertFalse(check.ok)
                self.assertIn("controls", check.detail)


if __name__ == "__main__":
    unittest.main()
