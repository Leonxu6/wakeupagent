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
