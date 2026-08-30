from maintenance.tempfile_mktemp_audit import audit_source

def test_secure_tempfile_is_allowed(): assert audit_source("tempfile.NamedTemporaryFile()\n")==[]
def test_mktemp_is_reported(): assert audit_source("tempfile.mktemp()\n")==["tempfile.mktemp() on line 1"]
