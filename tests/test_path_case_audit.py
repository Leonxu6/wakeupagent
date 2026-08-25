from pathlib import Path

from maintenance.path_case_audit import audit_paths


def test_path_case_audit_detects_cross_platform_collisions():
    failures = audit_paths([Path("docs/Guide.md"), Path("docs/guide.md")])
    assert failures == ["case-insensitive path collision: docs/Guide.md <-> docs/guide.md"]


def test_path_case_audit_accepts_distinct_paths():
    assert audit_paths([Path("docs/guide.md"), Path("tests/test_guide.py")]) == []
