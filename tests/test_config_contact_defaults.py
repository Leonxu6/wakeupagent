import importlib
import sys


def test_config_defaults_do_not_ship_private_contacts(monkeypatch):
    monkeypatch.delenv("WAKEUP_WECHAT_CONTACTS_JSON", raising=False)
    sys.modules.pop("config", None)
    config = importlib.import_module("config")
    assert config.WECHAT_CONTACTS == {}
