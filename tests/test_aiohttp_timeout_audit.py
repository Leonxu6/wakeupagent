from maintenance.aiohttp_timeout_audit import audit_source

def test_aiohttp_timeout_is_allowed(): assert audit_source("aiohttp.ClientSession(timeout=timeout)\n")==[]
def test_aiohttp_missing_timeout_is_reported(): assert audit_source("aiohttp.ClientSession()\n")==["aiohttp.ClientSession() without explicit timeout on line 1"]
