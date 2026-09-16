import os
from unittest.mock import patch

import pytest

from settings import env_bool


def test_boolean_parser_rejects_oversize_text_before_normalization():
    with patch.dict(os.environ, {"FLAG": "true" * 100_000}, clear=True):
        with pytest.raises(ValueError, match="boolean text"):
            env_bool("FLAG", False)


def test_boolean_parser_keeps_supported_spellings():
    for value, expected in (("TRUE", True), ("yes", True), ("0", False), ("OFF", False)):
        with patch.dict(os.environ, {"FLAG": value}, clear=True):
            assert env_bool("FLAG", not expected) is expected
