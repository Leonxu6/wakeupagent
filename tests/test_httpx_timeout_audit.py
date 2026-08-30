from maintenance.httpx_timeout_audit import audit_source

def test_httpx_timeout_is_allowed(): assert audit_source("httpx.get(url, timeout=10)\n")==[]
def test_httpx_missing_timeout_is_reported(): assert audit_source("httpx.AsyncClient()\n")==["httpx.AsyncClient() without explicit timeout on line 1"]
