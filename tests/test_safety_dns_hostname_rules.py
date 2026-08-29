import pytest

from safety import require_http_url


def test_browser_urls_reject_dns_labels_with_underscores():
    for url in ("https://bad_host.example/path", "http://service_bad.local:8080"):
        with pytest.raises(ValueError, match="hostname is malformed"):
            require_http_url(url)


def test_browser_urls_still_allow_hyphenated_dns_labels():
    assert require_http_url("https://good-host.example/path") == "https://good-host.example/path"
