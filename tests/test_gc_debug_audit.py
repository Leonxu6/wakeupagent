from maintenance.gc_debug_audit import audit_source

def test_allows_debug_lookup(): assert audit_source("gc.get_debug()\n")==[]
def test_reports_global_debug_mutation(): assert audit_source("gc.set_debug(gc.DEBUG_STATS)\n")==["gc.set_debug() mutates process-wide GC diagnostics on line 1"]
