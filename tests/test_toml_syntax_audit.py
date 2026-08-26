from pathlib import Path

from maintenance.toml_syntax_audit import audit_file


def test_toml_syntax_audit_accepts_valid_toml(tmp_path: Path):
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "demo"\n', encoding="utf-8")
    assert audit_file(path) == []


def test_toml_syntax_audit_reports_invalid_toml(tmp_path: Path):
    path = tmp_path / "broken.toml"
    path.write_text('[project\nname = "demo"\n', encoding="utf-8")
    failures = audit_file(path)
    assert len(failures) == 1
    assert "invalid TOML" in failures[0]
