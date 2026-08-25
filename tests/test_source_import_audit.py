from pathlib import Path

import maintenance.source_import_audit as audit_module


def test_source_import_audit_rejects_relative_imports(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_module, "_SOURCE_FILES", ("main.py",))
    (tmp_path / "main.py").write_text("from .local import thing\n", encoding="utf-8")
    failures = audit_module.audit(tmp_path)
    assert failures == ["main.py:1: relative imports are unsupported in the flat runtime layout"]


def test_source_import_audit_accepts_absolute_imports(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audit_module, "_SOURCE_FILES", ("main.py",))
    (tmp_path / "main.py").write_text("from pathlib import Path\n", encoding="utf-8")
    assert audit_module.audit(tmp_path) == []
