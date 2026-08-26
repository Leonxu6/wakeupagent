from diagnostics import Check, checks_payload, format_checks


def test_diagnostics_strip_control_characters_from_details():
    checks = [Check("camera", False, "backend\x00failed\x7fnow")]
    rendered = format_checks(checks)
    payload = checks_payload(checks)

    assert "backend failed now" in rendered
    assert payload[0]["detail"] == "backend failed now"
    assert "\x00" not in rendered
    assert "\x7f" not in rendered
