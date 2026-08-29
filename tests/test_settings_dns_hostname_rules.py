import pytest

from settings import env_http_url


def test_service_urls_reject_dns_labels_with_underscores(monkeypatch):
    monkeypatch.setenv("WA_TEST_URL", "https://bad_host.example/api")
    with pytest.raises(ValueError, match="hostname is malformed"):
        env_http_url("WA_TEST_URL", "https://fallback.example")


def test_service_urls_still_allow_hyphenated_dns_labels(monkeypatch):
    monkeypatch.setenv("WA_TEST_URL", "https://good-host.example/api")
    assert env_http_url("WA_TEST_URL", "https://fallback.example") == "https://good-host.example/api"
