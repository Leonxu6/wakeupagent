from types import SimpleNamespace

from diagnostics import _runtime_config


def test_runtime_config_reports_missing_required_fields(monkeypatch):
    monkeypatch.setattr("diagnostics.importlib.import_module", lambda _name: SimpleNamespace(CHECKPOINT_DB_PATH="state.db"))
    module, check = _runtime_config()
    assert module is None
    assert check.name == "configuration"
    assert not check.ok
    assert "DAILY_REPORT_PATH" in check.detail
    assert "OLLAMA_HOST" in check.detail


def test_runtime_config_accepts_complete_configuration_shape(monkeypatch):
    module = SimpleNamespace(
        CHECKPOINT_DB_PATH="state.db",
        DAILY_REPORT_PATH="report.txt",
        OLLAMA_HOST="http://127.0.0.1:11434",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        DEEPSEEK_API_KEY="",
    )
    monkeypatch.setattr("diagnostics.importlib.import_module", lambda _name: module)
    resolved, check = _runtime_config()
    assert resolved is module
    assert check.ok
    assert check.detail == "validated"
