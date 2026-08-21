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

    def test_collect_checks_is_side_effect_free_and_reports_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pose_landmarker_lite.task").write_bytes(b"pose")
            (root / "gesture_recognizer.task").write_bytes(b"gesture")
            with patch.object(diagnostics.config, "CHECKPOINT_DB_PATH", str(root / "state.db")), \
                 patch.object(diagnostics.config, "DAILY_REPORT_PATH", str(root / "reports.md")), \
                 patch.object(diagnostics.config, "OLLAMA_HOST", "http://localhost:11434"), \
                 patch.object(diagnostics.config, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"), \
                 patch.object(diagnostics.config, "DEEPSEEK_API_KEY", ""):
                checks = diagnostics.collect_checks(root)

        by_name = {check.name: check for check in checks}
        self.assertTrue(by_name["pose-model"].ok)
        self.assertTrue(by_name["gesture-model"].ok)
        self.assertTrue(by_name["checkpoint-dir"].ok)
        self.assertFalse(by_name["deepseek-key"].ok)

    def test_format_checks_has_stable_markers(self):
        text = diagnostics.format_checks([
            diagnostics.Check("a", True, "ready"),
            diagnostics.Check("b", False, "missing"),
        ])
        self.assertEqual(text, "[OK] a: ready\n[WARN] b: missing")

    def test_exit_code_only_fails_for_critical_models(self):
        self.assertEqual(diagnostics.diagnostics_exit_code([diagnostics.Check("deepseek-key", False, "missing")]), 0)
        self.assertEqual(diagnostics.diagnostics_exit_code([diagnostics.Check("pose-model", False, "missing")]), 1)


if __name__ == "__main__":
    unittest.main()
