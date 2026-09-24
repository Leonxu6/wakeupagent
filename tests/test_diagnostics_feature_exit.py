from diagnostics import Check, diagnostics_exit_code


def test_invalid_side_effect_flag_checks_fail_diagnostics():
    for name in ("tts", "browser-control", "external-messaging", "process-control"):
        assert diagnostics_exit_code([Check(name, False, "invalid boolean")]) == 1


def test_valid_disabled_side_effect_flags_keep_diagnostics_green():
    checks = [
        Check("tts", True, "disabled"),
        Check("browser-control", True, "disabled"),
        Check("external-messaging", True, "disabled"),
        Check("process-control", True, "disabled"),
    ]
    assert diagnostics_exit_code(checks) == 0


def test_optional_cloud_key_remains_noncritical():
    assert diagnostics_exit_code([Check("deepseek-key", False, "missing")]) == 0
