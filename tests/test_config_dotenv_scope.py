import importlib
import sys
from pathlib import Path

import dotenv


def _fresh_config():
    previous = sys.modules.pop("config", None)
    try:
        return importlib.import_module("config")
    finally:
        sys.modules.pop("config", None)
        if previous is not None:
            sys.modules["config"] = previous


def test_config_loads_only_project_dotenv_without_overriding_environment(monkeypatch):
    calls = []

    def fake_load_dotenv(*, dotenv_path=None, override=False, **kwargs):
        calls.append((dotenv_path, override, kwargs))
        return False

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    module = _fresh_config()
    expected = Path(module.__file__).resolve().with_name(".env")
    assert calls == [(expected, False, {})]


def test_relative_persistence_paths_are_anchored_to_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WAKEUP_CHECKPOINT_DB_PATH", "state/agent.db")
    monkeypatch.setenv("WAKEUP_DAILY_REPORT_PATH", "reports/daily.md")

    module = _fresh_config()
    root = Path(module.__file__).resolve().parent

    assert Path(module.CHECKPOINT_DB_PATH) == root / "state" / "agent.db"
    assert Path(module.DAILY_REPORT_PATH) == root / "reports" / "daily.md"


def test_absolute_persistence_paths_are_preserved(monkeypatch, tmp_path):
    checkpoint = tmp_path / "agent.db"
    report = tmp_path / "daily.md"
    monkeypatch.setenv("WAKEUP_CHECKPOINT_DB_PATH", str(checkpoint))
    monkeypatch.setenv("WAKEUP_DAILY_REPORT_PATH", str(report))

    module = _fresh_config()

    assert Path(module.CHECKPOINT_DB_PATH) == checkpoint
    assert Path(module.DAILY_REPORT_PATH) == report
