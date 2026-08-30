from maintenance.zip_extractall_audit import audit_source

def test_archive_inspection_is_allowed(): assert audit_source("archive.infolist()\n")==[]
def test_extractall_is_reported(): assert audit_source("archive.extractall(target)\n")==["extractall() needs traversal review on line 1"]
