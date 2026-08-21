import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_model_check_reports_missing_empty_and_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = diagnostics._model_check("model", root / "missing.task")
            self.assertFalse(missing.ok)
            empty_path = root / "empty.task"
            empty_path.write_bytes(b"")
            empty = diagnostics._model_check("model", empty_path)
            self.assertFalse(empty.ok)
            valid_path = root / "valid.task"
            valid_path.write_bytes(b"model")
            valid = diagnostics._model_check("model", valid_path)
            self.assertTrue(valid.ok)

    def test_model_check_reports_unreadable_files_without_crashing(self):
        with patch.object(Path, "stat", side_effect=PermissionError("denied")):
            check = diagnostics._model_check("model", Path("model.task"))
        self.assertFalse(check.ok)
        self.assertIn("unreadable", check.detail)
        self.assertIn("denied", check.detail)

    def test_directory_check_rejects_missing_and_non_directory_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = diagnostics._directory_check("state", root / "missing")
            self.assertFalse(missing.ok)
            self.assertIn("missing", missing.detail)

            regular_file = root / "state"
            regular_file.write_text("not a directory", encoding="utf-8")
            not_directory = diagnostics._directory_check("state", regular_file)
            self.assertFalse(not_directory.ok)
            self.assertIn("not a directory", not_directory.detail)

            valid = diagnostics._directory_check("state", root)
            self.assertTrue(valid.ok)

    def test_directory_check_reports_unreadable_metadata_without_crashing(self):
        with patch.object(Path, "stat", side_effect=PermissionError("blocked")):
            check = diagnostics._directory_check("state", Path("memory"))
        self.assertFalse(check.ok)
        self.assertIn("unreadable", check.detail)
        self.assertIn("blocked", check.detail)

    def test_http_url_check_matches_runtime_url_boundaries(self):
        for value in ("ftp://example.com", "https://user:secret@example.com", "https://example.com:bad"):
            with self.subTest(value=value):
                self.assertFalse(diagnostics._http_url_check("service", value).ok)

        valid = diagnostics._http_url_check("service", "https://example.com:8443/api")
        self.assertTrue(valid.ok)
        self.assertEqual(valid.detail, "https://example.com:8443/api")

    def test_feature_flag_check_reports_enabled_disabled_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            disabled = diagnostics._feature_flag_check("feature", "FEATURE")
        self.assertTrue(disabled.ok)
        self.assertEqual(disabled.detail, "disabled")

        with patch.dict(os.environ, {"FEATURE": "true"}, clear=True):
            enabled = diagnostics._feature_flag_check("feature", "FEATURE")
        self.assertTrue(enabled.ok)
        self.assertEqual(enabled.detail, "enabled")

        with patch.dict(os.environ, {"FEATURE": "sometimes"}, clear=True):
            invalid = diagnostics._feature_flag_check("feature", "FEATURE")
        self.assertFalse(invalid.ok)
        self.assertIn("must be one of", invalid.detail)

    def test_collect_checks_is_side_effect_free_and_reports_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pose_landmarker_lite.task").write_bytes(b"pose")
            (root / "gesture_recognizer.task").write_bytes(b"gesture")
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(diagnostics.config, "CHECKPOINT_DB_PATH", str(root / "state.db")), \
                 patch.object(diagnostics.config, "DAILY_REPORT_PATH", str(root / "reports.md")), \
                 patch.object(diagnostics.config, "OLLAMA_HOST", "http://localhost:11434"), \
                 patch.object(diagnostics.config, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"), \
                 patch.object(diagnostics.config, "DEEPSEEK_API_KEY", ""):
                checks = diagnostics.collect_checks(root)

        by_name = {check.name: check for check in checks}
        self.assertTrue(by_name["pose-model"].ok)
        self.assertTrue(by_name["gesture-model"].ok)
        self.assertTrue(by_name["checkpoint-dir"].ok)
        self.assertTrue(by_name["ollama-url"].ok)
        self.assertTrue(by_name["deepseek-url"].ok)
        self.assertFalse(by_name["deepseek-key"].ok)
        self.assertEqual(by_name["external-messaging"].detail, "disabled")
        self.assertEqual(by_name["process-control"].detail, "disabled")

    def test_format_checks_has_stable_markers_and_one_line_per_check(self):
        text = diagnostics.format_checks([
            diagnostics.Check("a", True, "ready"),
            diagnostics.Check("b\nspoof", False, "missing\n[OK] forged: yes"),
        ])
        self.assertEqual(text, "[OK] a: ready\n[WARN] b spoof: missing [OK] forged: yes")
        self.assertEqual(len(text.splitlines()), 2)

    def test_exit_code_only_fails_for_critical_models(self):
        self.assertEqual(diagnostics.diagnostics_exit_code([diagnostics.Check("deepseek-key", False, "missing")]), 0)
        self.assertEqual(diagnostics.diagnostics_exit_code([diagnostics.Check("pose-model", False, "missing")]), 1)


if __name__ == "__main__":
    unittest.main()
