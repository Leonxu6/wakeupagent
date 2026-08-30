from maintenance.contextlib_suppress_audit import audit_source

def test_specific_suppression_is_allowed(): assert audit_source("with contextlib.suppress(FileNotFoundError):\n    pass\n")==[]
def test_broad_suppression_is_reported(): assert audit_source("with contextlib.suppress(Exception):\n    pass\n")==["broad contextlib.suppress() on line 1"]
