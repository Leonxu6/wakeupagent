from pathlib import Path

from maintenance.text_integrity_audit import audit


def test_text_integrity_audit_accepts_clean_utf8(tmp_path: Path):
    (tmp_path / "README.md").write_text("hello 世界\n", encoding="utf-8")
    assert audit(tmp_path) == []


def test_text_integrity_audit_reports_bom_nul_and_missing_newline(tmp_path: Path):
    (tmp_path / "bom.md").write_bytes(b"\xef\xbb\xbfhello\n")
    (tmp_path / "nul.py").write_bytes(b"x = 'a\x00b'\n")
    (tmp_path / "tail.toml").write_text("x = 1", encoding="utf-8")
    failures = audit(tmp_path)
    assert any("BOM" in item for item in failures)
    assert any("NUL" in item for item in failures)
    assert any("final newline" in item for item in failures)
