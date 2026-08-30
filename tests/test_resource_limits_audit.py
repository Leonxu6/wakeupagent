from maintenance.resource_limits_audit import audit_source

def test_limit_reads_are_allowed(): assert audit_source("resource.getrlimit(resource.RLIMIT_NOFILE)\n")==[]
def test_limit_mutation_is_reported(): assert audit_source("resource.setrlimit(resource.RLIMIT_NOFILE, limits)\n")==["resource.setrlimit() mutates process limits on line 1"]
