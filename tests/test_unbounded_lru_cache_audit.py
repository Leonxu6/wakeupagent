from maintenance.unbounded_lru_cache_audit import audit_source

def test_bounded_cache_is_allowed(): assert audit_source("functools.lru_cache(maxsize=256)()\n")==[]
def test_unbounded_cache_is_reported(): assert audit_source("functools.lru_cache(maxsize=None)()\n")==["unbounded functools.lru_cache() on line 1"]
