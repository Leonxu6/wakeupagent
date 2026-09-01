from maintenance.sys_int_max_str_digits_audit import audit_source

def test_allows_regular_integer_use(): assert audit_source("value = int(text)\n")==[]
def test_reports_global_limit_mutation(): assert audit_source("sys.set_int_max_str_digits(0)\n")==["sys.set_int_max_str_digits() mutates a process-wide conversion limit on line 1"]
