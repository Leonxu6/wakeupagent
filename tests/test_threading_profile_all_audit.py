from maintenance.threading_profile_all_audit import audit_source

def test_allows_profile_lookup(): assert audit_source("threading.getprofile()\n")==[]
def test_reports_global_profile_installation(): assert audit_source("threading.setprofile_all_threads(profile)\n")==["threading.setprofile_all_threads() mutates global profiling behavior on line 1"]
