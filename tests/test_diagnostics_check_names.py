import pytest

from diagnostics import Check, checks_payload, diagnostics_exit_code, format_checks


def test_diagnostic_check_names_reject_padding_controls_and_oversize():
    for name in ("", " python", "python ", "python\n", "x" * 81):
        with pytest.raises(ValueError):
            format_checks([Check(name, True, "ok")])


def test_diagnostic_check_names_still_reject_casefold_collisions():
    with pytest.raises(ValueError, match="duplicate"):
        checks_payload([Check("Python", True, "ok"), Check("python", True, "ok")])


def test_critical_exit_code_uses_validated_exact_identifier():
    assert diagnostics_exit_code([Check("python", False, "too old")]) == 1
    with pytest.raises(ValueError):
        diagnostics_exit_code([Check(" python ", False, "ambiguous")])


def test_payload_preserves_clean_identifiers_and_sanitizes_details():
    payload = checks_payload([Check("configuration", True, "good\nvalue")])
    assert payload == [{"name": "configuration", "ok": True, "detail": "good value"}]
