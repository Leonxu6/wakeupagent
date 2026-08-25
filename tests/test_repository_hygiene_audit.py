from pathlib import Path

from maintenance.repository_hygiene_audit import audit_paths


def test_repository_hygiene_rejects_private_and_generated_paths():
    failures = audit_paths([
        Path(".env"),
        Path("__pycache__/module.pyc"),
        Path("build/cache.pyo"),
        Path("src/main.py"),
    ])
    assert len(failures) == 3
    assert all("tracked" in item for item in failures)


def test_repository_hygiene_accepts_source_and_docs():
    assert audit_paths([Path("main.py"), Path("docs/privacy.md")]) == []
