from maintenance.sys_trace_audit import audit_source

def test_allows_trace_lookup(): assert audit_source("sys.gettrace()\n")==[]
def test_reports_trace_installation(): assert audit_source("sys.settrace(trace)\n")==["sys.settrace() installs a runtime tracing hook on line 1"]
