import os
from unittest.mock import patch

import pytest

from settings import env_http_url


def test_service_url_rejects_encoded_slash_separators():
    for value in (
        "https://example.com/api%2Fv1",
        "https://example.com/api%2fv1",
        "https://example.com/a/%2F/b",
    ):
        with patch.dict(os.environ, {"URL": value}, clear=True):
            with pytest.raises(ValueError, match="encoded slash"):
                env_http_url("URL", "http://localhost")


def test_service_url_preserves_non_separator_percent_encoding():
    with patch.dict(os.environ, {"URL": "https://example.com/model%20api"}, clear=True):
        assert env_http_url("URL", "http://localhost") == "https://example.com/model%20api"
