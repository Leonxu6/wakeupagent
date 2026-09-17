import json
from pathlib import Path

from maintenance.json_syntax_audit import audit_file


def test_json_syntax_audit_accepts_valid_json(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    assert audit_file(path) == []


def test_json_syntax_audit_reports_invalid_json(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text('{"enabled":', encoding="utf-8")
    failures = audit_file(path)
    assert len(failures) == 1
    assert "invalid JSON" in failures[0]


def test_json_syntax_audit_rejects_nonstandard_constants(tmp_path: Path):
    for constant in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / f"{constant.replace('-', 'neg-')}.json"
        path.write_text('{"value":' + constant + "}", encoding="utf-8")
        failures = audit_file(path)
        assert len(failures) == 1
        assert "invalid JSON" in failures[0]


def test_json_syntax_audit_rejects_duplicate_object_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"mode":"safe","mode":"unsafe"}', encoding="utf-8")
    failures = audit_file(path)
    assert len(failures) == 1
    assert "invalid JSON" in failures[0]
    assert "duplicate" in failures[0]


def test_json_syntax_audit_rejects_symlinked_json(tmp_path: Path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"enabled":true}', encoding="utf-8")
    link = tmp_path / "config.json"
    link.symlink_to(outside)

    failures = audit_file(link)

    assert len(failures) == 1
    assert "symbolic links" in failures[0]


def test_json_syntax_audit_rejects_directory_named_json(tmp_path: Path):
    path = tmp_path / "config.json"
    path.mkdir()

    failures = audit_file(path)

    assert len(failures) == 1
    assert "regular file" in failures[0]
