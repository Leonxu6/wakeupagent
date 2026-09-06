import unittest
from pathlib import Path

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

    def test_path_objects_share_the_same_control_boundary(self):
        for value in (Path("memory\u200c/state.db"), Path("memory/report\u206a.md")):
            with self.subTest(value=value):
                check = diagnostics._persistence_parent_check("state", value)
                self.assertFalse(check.ok)
                self.assertIn("controls", check.detail)

    def test_diagnostic_roots_reject_hidden_controls_before_resolution(self):
        for value in ("root\u200c", Path("root\u206a")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "controls"):
                    diagnostics._diagnostic_root(value)


if __name__ == "__main__":
    unittest.main()
