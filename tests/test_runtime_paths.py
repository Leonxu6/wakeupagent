import unittest
from pathlib import Path
from unittest.mock import patch

import runtime_paths


class RuntimePathTests(unittest.TestCase):
    def test_resolves_relative_paths_from_project_root(self):
        with patch.object(runtime_paths, "PROJECT_ROOT", Path("/tmp/wakeupagent")):
            self.assertEqual(
                runtime_paths.resolve_runtime_path("memory/report.md"),
                Path("/tmp/wakeupagent/memory/report.md"),
            )

    def test_preserves_absolute_paths(self):
        path = Path("/var/tmp/wakeupagent.db")
        self.assertEqual(runtime_paths.resolve_runtime_path(path), path)

    def test_expands_user_home(self):
        with patch("pathlib.Path.expanduser", return_value=Path("/home/test/report.md")):
            self.assertEqual(
                runtime_paths.resolve_runtime_path("~/report.md"),
                Path("/home/test/report.md"),
            )

    def test_rejects_blank_string_paths(self):
        for value in ("", " ", "\t"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    runtime_paths.resolve_runtime_path(value)

    def test_rejects_null_bytes(self):
        with self.assertRaisesRegex(ValueError, "null bytes"):
            runtime_paths.resolve_runtime_path("memory/\x00report.md")


if __name__ == "__main__":
    unittest.main()
