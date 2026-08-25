from pathlib import Path

from maintenance.readme_command_audit import audit


def test_readme_command_audit_accepts_supported_commands(tmp_path: Path):
    (tmp_path / "README.md").write_text("uv sync\nuv run main.py --check\nuv run main.py --check-json\n", encoding="utf-8")
    assert audit(tmp_path) == []


def test_readme_command_audit_reports_missing_diagnostics(tmp_path: Path):
    (tmp_path / "README.md").write_text("uv sync\n", encoding="utf-8")
    failures = audit(tmp_path)
    assert len(failures) == 2
    assert any("--check-json" in item for item in failures)
