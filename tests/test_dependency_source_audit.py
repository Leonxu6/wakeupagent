from maintenance.dependency_source_audit import audit_dependencies


def test_dependency_source_audit_accepts_index_constraints():
    assert audit_dependencies(["httpx>=0.27", "rich>=13"]) == []


def test_dependency_source_audit_reports_direct_urls():
    failures = audit_dependencies(["demo @ https://example.com/demo.whl"])
    assert failures == ["direct dependency source is not allowed: demo @ https://example.com/demo.whl"]
