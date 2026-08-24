import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import diagnostics


class DiagnosticDirectoryPermissionTests(unittest.TestCase):
    def test_directory_check_reports_unwritable_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with patch("diagnostics.os.access", return_value=False):
                check = diagnostics._directory_check("state", path)
        self.assertFalse(check.ok)
        self.assertIn("not writable", check.detail)

    def test_directory_check_keeps_writable_paths_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with patch("diagnostics.os.access", return_value=True):
                check = diagnostics._directory_check("state", path)
        self.assertTrue(check.ok)


if __name__ == "__main__":
    unittest.main()
