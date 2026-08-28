from maintenance.naive_datetime_now_audit import audit_source


def test_datetime_now_allows_timezone():
    assert audit_source("datetime.now(timezone.utc)\n") == []


def test_datetime_now_reports_naive_call():
    assert audit_source("datetime.now()\n") == ["datetime.now() without timezone on line 1"]
