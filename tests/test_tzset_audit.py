from maintenance.tzset_audit import audit_source

def test_timezone_reads_are_allowed(): assert audit_source("time.localtime()\n")==[]
def test_tzset_is_reported(): assert audit_source("time.tzset()\n")==["time.tzset() mutates process timezone on line 1"]
