from maintenance.unicode_bidi_audit import audit_text


def test_unicode_bidi_audit_accepts_regular_unicode():
    assert audit_text("hello 世界\n") == []


def test_unicode_bidi_audit_reports_direction_override():
    assert audit_text("safe\u202etxt") == ["bidirectional control character on line 1"]
