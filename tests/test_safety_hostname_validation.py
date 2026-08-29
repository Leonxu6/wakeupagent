import pytest

from safety import require_http_url


@pytest.mark.parametrize(
    "url",
    [
        "https://.",
        "https://example..com",
        "https://-example.com",
        "https://example-.com",
        "https://example.com.",
    ],
)
def test_http_urls_reject_structurally_invalid_hostnames(url):
    with pytest.raises(ValueError, match="hostname"):
        require_http_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "https://example.com/path?q=1#section",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "http://service_name:8080",
    ],
)
def test_http_urls_keep_supported_local_and_dns_hosts(url):
    assert require_http_url(url) == url
