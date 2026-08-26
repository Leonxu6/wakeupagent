import pytest

from tools import _observation_text


def test_observation_text_rejects_nul_from_model_output():
    with pytest.raises(ValueError, match="control characters"):
        _observation_text("person studying\x00hidden")


def test_observation_text_normalizes_regular_whitespace():
    assert _observation_text("person   studying") == "person studying"
