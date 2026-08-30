from maintenance.os_putenv_audit import audit_source

def test_environment_reads_are_allowed(): assert audit_source("os.getenv('MODE')\n")==[]
def test_putenv_is_reported(): assert audit_source("os.putenv('MODE','prod')\n")==["os.putenv() mutates process environment on line 1"]
