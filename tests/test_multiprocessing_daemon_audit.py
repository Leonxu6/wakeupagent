from maintenance.multiprocessing_daemon_audit import audit_source

def test_explicit_daemon_policy_is_allowed(): assert audit_source("multiprocessing.Process(target=worker, daemon=False)\n")==[]
def test_implicit_daemon_policy_is_reported(): assert audit_source("multiprocessing.Process(target=worker)\n")==["multiprocessing.Process() without explicit daemon policy on line 1"]
