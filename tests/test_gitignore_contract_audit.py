from pathlib import Path

import maintenance.gitignore_contract_audit as audit_module


def test_gitignore_contract_reports_missing_rules(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_module, "_REQUIRED", {".env", ".venv"})
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    assert audit_module.audit(tmp_path) == [".gitignore: missing required privacy/generated rule .venv"]


def test_gitignore_contract_ignores_comments_and_blank_lines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_module, "_REQUIRED", {".env"})
    (tmp_path / ".gitignore").write_text("# private\n\n.env\n", encoding="utf-8")
    assert audit_module.audit(tmp_path) == []
