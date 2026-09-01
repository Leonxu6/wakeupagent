from maintenance.threading_stack_size_audit import audit_source

def test_allows_reading_default(): assert audit_source("threading.stack_size()\n")==[]
def test_reports_default_stack_mutation(): assert audit_source("threading.stack_size(1048576)\n")==["threading.stack_size() mutates the default thread stack on line 1"]
