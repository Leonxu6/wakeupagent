import diagnostics


def test_runtime_config_redacts_import_error_details(monkeypatch):
    def fail_import(name):
        raise RuntimeError("api_key=secret path=/Users/private/config.py")

    monkeypatch.setattr(diagnostics.importlib, "import_module", fail_import)

    module, check = diagnostics._runtime_config()

    assert module is None
    assert check.ok is False
    assert check.detail == "configuration import failed (RuntimeError)"
    assert "secret" not in check.detail
    assert "/Users" not in check.detail
