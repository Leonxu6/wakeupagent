from pathlib import Path

import maintenance.docs_contract_audit as audit_module


def test_docs_contract_reports_missing_documents(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_module, "_REQUIRED", {"README.md": 5, "docs/testing.md": 5})
    (tmp_path / "README.md").write_text("enough\n", encoding="utf-8")
    failures = audit_module.audit(tmp_path)
    assert failures == ["docs/testing.md: required maintainer documentation is missing"]


def test_docs_contract_reports_unexpectedly_small_documents(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_module, "_REQUIRED", {"README.md": 20})
    (tmp_path / "README.md").write_text("tiny\n", encoding="utf-8")
    assert "unexpectedly small" in audit_module.audit(tmp_path)[0]
