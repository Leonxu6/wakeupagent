from pathlib import Path

from maintenance.env_parity_audit import audit, template_env_names


def _runtime_files(root: Path):
    (root / "config.py").write_text('VALUE = env_text("EXAMPLE_VALUE", "x")\n', encoding="utf-8")
    (root / "diagnostics.py").write_text('FLAG = env_bool("EXAMPLE_FLAG", False)\n', encoding="utf-8")
    (root / "tools.py").write_text('_feature_enabled("EXAMPLE_SIDE_EFFECT")\n', encoding="utf-8")


def test_template_env_names_ignores_comments():
    assert template_env_names("# HIDDEN=x\nVISIBLE=y\n") == {"VISIBLE"}


def test_env_parity_reports_missing_runtime_variables(tmp_path: Path):
    _runtime_files(tmp_path)
    (tmp_path / ".env.example").write_text("EXAMPLE_VALUE=x\nEXAMPLE_FLAG=false\n", encoding="utf-8")
    failures = audit(tmp_path)
    assert failures == [".env.example: missing runtime variable EXAMPLE_SIDE_EFFECT"]
