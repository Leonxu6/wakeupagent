from maintenance.sqlite_callback_tracebacks_audit import audit_source

def test_allows_regular_connections(): assert audit_source("sqlite3.connect(path,timeout=5)\n")==[]
def test_reports_global_diagnostic_changes(): assert audit_source("sqlite3.enable_callback_tracebacks(True)\n")==["sqlite3.enable_callback_tracebacks() mutates process diagnostics on line 1"]
