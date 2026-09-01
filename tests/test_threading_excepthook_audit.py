from maintenance.threading_excepthook_audit import audit_source

def test_allows_local_handlers(): assert audit_source("handler = report_error\n")==[]
def test_reports_global_hook_replacement(): assert audit_source("threading.excepthook = report_error\n")==["threading.excepthook replacement mutates thread error handling on line 1"]
