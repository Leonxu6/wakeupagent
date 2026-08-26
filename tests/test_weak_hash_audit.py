from maintenance.weak_hash_audit import audit_source


def test_weak_hash_audit_accepts_sha256():
    assert audit_source("hashlib.sha256(data).digest()\n") == []


def test_weak_hash_audit_reports_md5_and_sha1():
    failures = audit_source("hashlib.md5(data)\nhashlib.sha1(data)\n")
    assert failures == [
        "weak hash hashlib.md5() on line 1",
        "weak hash hashlib.sha1() on line 2",
    ]
