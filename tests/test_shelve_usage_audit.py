from maintenance.shelve_usage_audit import audit_source

def test_json_persistence_is_allowed(): assert audit_source("json.dump(data, handle)\n")==[]
def test_shelve_persistence_is_reported(): assert audit_source("shelve.open(path)\n")==["shelve.open() uses pickle-backed persistence on line 1"]
