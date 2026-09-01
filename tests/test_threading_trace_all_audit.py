from maintenance.threading_trace_all_audit import audit_source

def test_allows_trace_lookup(): assert audit_source("threading.gettrace()\n")==[]
def test_reports_global_trace_installation(): assert audit_source("threading.settrace_all_threads(trace)\n")==["threading.settrace_all_threads() mutates global tracing behavior on line 1"]
