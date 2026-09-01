from maintenance.gc_threshold_audit import audit_source

def test_allows_threshold_lookup(): assert audit_source("gc.get_threshold()\n")==[]
def test_reports_collection_policy_mutation(): assert audit_source("gc.set_threshold(700,10,10)\n")==["gc.set_threshold() mutates process-wide collection policy on line 1"]
