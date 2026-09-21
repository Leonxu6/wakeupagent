from pathlib import Path

from maintenance.dependency_lock_audit import _MAX_LOCK_BYTES, audit


def _write_project(root: Path, requirement: str = ">=3.12") -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "wakeupagent"\nrequires-python = "{requirement}"\n',
        encoding="utf-8",
    )


def _write_lock(root: Path, requirement: str = ">=3.12") -> None:
    (root / "uv.lock").write_text(
        f'requires-python = "{requirement}"\n[[package]]\nname = "demo"\nversion = "1.0"\n',
        encoding="utf-8",
    )


def test_dependency_lock_accepts_matching_python_requirement(tmp_path: Path):
    _write_project(tmp_path)
    _write_lock(tmp_path)
    assert audit(tmp_path) == []


def test_dependency_lock_reports_python_drift(tmp_path: Path):
    _write_project(tmp_path)
    _write_lock(tmp_path, ">=3.11")
    failures = audit(tmp_path)
    assert any("does not match" in item for item in failures)


def test_dependency_lock_rejects_malformed_toml(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / "uv.lock").write_text('requires-python = ">=3.12"\n[[package]\n', encoding="utf-8")
    failures = audit(tmp_path)
    assert any("could not parse uv.lock" in item for item in failures)


def test_dependency_lock_rejects_missing_package_tables(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / "uv.lock").write_text('requires-python = ">=3.12"\n', encoding="utf-8")
    failures = audit(tmp_path)
    assert any("package table is unexpectedly incomplete" in item for item in failures)


def test_dependency_lock_rejects_symlinked_lockfile(tmp_path: Path):
    _write_project(tmp_path)
    target = tmp_path / "real.lock"
    target.write_text('requires-python = ">=3.12"\n[[package]]\nname = "demo"\n', encoding="utf-8")
    (tmp_path / "uv.lock").symlink_to(target)
    failures = audit(tmp_path)
    assert any("must not be a symbolic link" in item for item in failures)


def test_dependency_lock_rejects_oversized_lockfile(tmp_path: Path):
    _write_project(tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"x" * (_MAX_LOCK_BYTES + 1))
    failures = audit(tmp_path)
    assert any("exceeds" in item for item in failures)
