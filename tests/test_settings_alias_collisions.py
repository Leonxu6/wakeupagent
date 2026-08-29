import pytest

from settings import env_json_string_map


def test_json_map_rejects_case_insensitive_alias_collisions(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_MAP", '{"Family":"Mom","family":"Dad"}')
    with pytest.raises(ValueError, match="duplicate"):
        env_json_string_map("WAKEUP_TEST_MAP", {})


def test_json_map_rejects_unicode_normalization_collisions(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_MAP", '{"Ａlice":"One","Alice":"Two"}')
    with pytest.raises(ValueError, match="normalization-equivalent"):
        env_json_string_map("WAKEUP_TEST_MAP", {})


def test_json_map_accepts_distinct_aliases(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_MAP", '{"family":"Mom","mentor":"Teacher"}')
    assert env_json_string_map("WAKEUP_TEST_MAP", {}) == {"family": "Mom", "mentor": "Teacher"}
