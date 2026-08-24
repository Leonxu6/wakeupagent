import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import diagnostics


class DiagnosticRootTests(unittest.TestCase):
    def test_accepts_path_objects_and_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(diagnostics._diagnostic_root(root), root)
            self.assertEqual(diagnostics._diagnostic_root(str(root)), root)

    def test_rejects_accidental_scalar_roots(self):
        for value in (0, False, 3.14, object()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostics._diagnostic_root(value)

    def test_none_uses_module_directory(self):
        with patch.object(diagnostics.Path, "resolve", return_value=Path("/tmp/wakeupagent/diagnostics.py")):
            self.assertEqual(diagnostics._diagnostic_root(None), Path("/tmp/wakeupagent"))


if __name__ == "__main__":
    unittest.main()
