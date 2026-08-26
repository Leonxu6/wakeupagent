from maintenance.tempfile_safety_audit import audit_source


def test_tempfile_safety_audit_accepts_named_temporary_file():
    assert audit_source("tempfile.NamedTemporaryFile()\n") == []


def test_tempfile_safety_audit_reports_mktemp():
    assert audit_source("tempfile.mktemp()\n") == ["insecure tempfile.mktemp() on line 1"]
