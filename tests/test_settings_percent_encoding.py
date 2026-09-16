import os
from unittest.mock import patch

import pytest

from settings import env_http_url


def test_http_url_rejects_malformed_percent_encoding():
    invalid = (
        "https://example.com/api/%",
        "https://example.com/api/%2",
        "https://example.com/api/%GG",
        "https://example.com/api/%FF",
    )
    for value in invalid:
        with patch.dict(os.environ, {"URL": value}, clear=True):
            with pytest.raises(ValueError, match="percent encoding"):
                env_http_url("URL", "http://localhost")


def test_http_url_keeps_valid_utf8_percent_encoding():
    value = "https://example.com/%E4%B8%AD"
    with patch.dict(os.environ, {"URL": value}, clear=True):
        assert env_http_url("URL", "http://localhost") == value
