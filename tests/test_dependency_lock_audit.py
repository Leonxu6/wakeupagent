from pathlib import Path

from maintenance.dependency_lock_audit import audit


def test_dependency_lock_accepts_matching_python_requirement(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('requires-python = ">=3.12"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text('requires-python = ">=3.12"\n' + ('x' * 1000) + '\n[[package]]\nname="demo"\n', encoding="utf-8")
    assert audit(tmp_path) == []


def test_dependency_lock_reports_python_drift(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('requires-python = ">=3.12"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text('requires-python = ">=3.11"\n', encoding="utf-8")
    failures = audit(tmp_path)
    assert any("does not match" in item for item in failures)
    assert any("incomplete" in item for item in failures)
