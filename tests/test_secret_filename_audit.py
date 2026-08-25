from pathlib import Path

from maintenance.secret_filename_audit import audit_paths


def test_secret_filename_audit_flags_key_material():
    failures = audit_paths([Path("certs/client.pem"), Path("credentials.json"), Path("src/main.py")])
    assert len(failures) == 2


def test_secret_filename_audit_accepts_templates():
    assert audit_paths([Path(".env.example"), Path("docs/security.md")]) == []
