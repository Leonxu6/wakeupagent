import pytest

from settings import env_json_string_map


def test_json_map_parser_caps_configured_entry_limits(monkeypatch):
    monkeypatch.delenv("WAKEUP_TEST_MAP", raising=False)
    assert env_json_string_map("WAKEUP_TEST_MAP", {}, max_entries=1000) == {}
    with pytest.raises(ValueError, match="1000"):
        env_json_string_map("WAKEUP_TEST_MAP", {}, max_entries=1001)
