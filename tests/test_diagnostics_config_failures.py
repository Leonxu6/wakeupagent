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
    checks = [Check("configuration", False, "invalid")]
    assert diagnostics_exit_code(checks) == 1


def test_valid_configuration_check_does_not_fail_exit_code():
    checks = [Check("configuration", True, "validated")]
    assert diagnostics_exit_code(checks) == 0
