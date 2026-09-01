from maintenance.sqlite_shared_cache_audit import audit_source

def test_allows_connection_options(): assert audit_source("sqlite3.connect(path,uri=True)\n")==[]
def test_reports_global_policy_changes(): assert audit_source("sqlite3.enable_shared_cache(True)\n")==["sqlite3.enable_shared_cache() mutates global SQLite connection policy on line 1"]
