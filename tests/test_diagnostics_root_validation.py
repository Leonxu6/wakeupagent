import tempfile
import unittest
from pathlib import Path

import diagnostics


class DiagnosticRootValidationTests(unittest.TestCase):
    def test_explicit_root_must_be_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(diagnostics._diagnostic_root(root), root.resolve())
            file_path = root / "config.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "existing directory"):
                diagnostics._diagnostic_root(file_path)
            with self.assertRaisesRegex(ValueError, "existing directory"):
                diagnostics._diagnostic_root(root / "missing")


if __name__ == "__main__":
    unittest.main()
