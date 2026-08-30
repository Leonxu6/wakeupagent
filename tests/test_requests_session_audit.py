from maintenance.requests_session_audit import audit_source

def test_stateless_request_is_allowed(): assert audit_source("requests.get(url, timeout=10)\n")==[]
def test_session_creation_is_reported(): assert audit_source("requests.Session()\n")==["requests.Session() needs explicit lifecycle on line 1"]
