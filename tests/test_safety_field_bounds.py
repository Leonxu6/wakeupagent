import pytest

from safety import require_text


def test_validation_field_labels_are_bounded():
    with pytest.raises(ValueError, match="field must be at most 80"):
        require_text("ok", field="f" * 81, max_length=10)


def test_normal_field_labels_remain_supported():
    assert require_text("ok", field="camera description", max_length=10) == "ok"
