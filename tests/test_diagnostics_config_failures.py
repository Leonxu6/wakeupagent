import diagnostics
from diagnostics import Check, diagnostics_exit_code


def test_runtime_config_reports_validation_failures(monkeypatch):
    def fail_import(name):
        assert name == "config"
        raise ValueError("WAKEUP_CAPTURE_INTERVAL_SEC must be a number")

    monkeypatch.setattr(diagnostics.importlib, "import_module", fail_import)
    module, check = diagnostics._runtime_config()
    assert module is None
    assert check == Check(
        "configuration",
        False,
        "WAKEUP_CAPTURE_INTERVAL_SEC must be a number",
    )


def test_configuration_failure_is_a_critical_diagnostic():
    assert diagnostics_exit_code([Check("configuration", False, "invalid")]) == 1


def test_persistence_failures_are_critical_diagnostics():
    assert diagnostics_exit_code([Check("checkpoint-dir", False, "not writable")]) == 1
    assert diagnostics_exit_code([Check("report-dir", False, "missing")]) == 1


def test_missing_deepseek_key_is_a_critical_diagnostic():
    assert diagnostics_exit_code([Check("deepseek-key", False, "not configured")]) == 1


def test_configured_deepseek_key_does_not_fail_exit_code():
    assert diagnostics_exit_code([Check("deepseek-key", True, "configured")]) == 0


def test_noncritical_warning_does_not_fail_exit_code():
    assert diagnostics_exit_code([Check("platform", False, "Linux")]) == 0


def test_valid_configuration_check_does_not_fail_exit_code():
    assert diagnostics_exit_code([Check("configuration", True, "validated")]) == 0
