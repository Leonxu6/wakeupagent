import importlib
import sys
from pathlib import Path

import dotenv


def test_config_loads_only_project_dotenv_without_overriding_environment(monkeypatch):
    calls = []

    def fake_load_dotenv(*, dotenv_path=None, override=False, **kwargs):
        calls.append((dotenv_path, override, kwargs))
        return False

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    previous = sys.modules.pop("config", None)
    try:
        module = importlib.import_module("config")
        expected = Path(module.__file__).with_name(".env")
        assert calls == [(expected, False, {})]
    finally:
        sys.modules.pop("config", None)
        if previous is not None:
            sys.modules["config"] = previous
