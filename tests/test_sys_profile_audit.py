from maintenance.sys_profile_audit import audit_source

def test_allows_profile_lookup(): assert audit_source("sys.getprofile()\n")==[]
def test_reports_profile_installation(): assert audit_source("sys.setprofile(profile)\n")==["sys.setprofile() installs a runtime profiling hook on line 1"]
