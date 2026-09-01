from maintenance.atexit_register_audit import audit_source

def test_allows_local_cleanup_calls(): assert audit_source("cleanup()\n")==[]
def test_reports_process_shutdown_hooks(): assert audit_source("atexit.register(cleanup)\n")==["atexit.register() mutates process shutdown behavior on line 1"]
