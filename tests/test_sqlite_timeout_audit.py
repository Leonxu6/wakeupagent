from maintenance.sqlite_timeout_audit import audit_source

def test_explicit_timeout_is_allowed(): assert audit_source("sqlite3.connect(path, timeout=10)\n")==[]
def test_implicit_timeout_is_reported(): assert audit_source("sqlite3.connect(path)\n")==["sqlite3.connect() without explicit timeout on line 1"]
