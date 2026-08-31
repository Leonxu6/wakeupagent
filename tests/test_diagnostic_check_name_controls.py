import pytest

from diagnostics import _check_name


def test_check_name_rejects_control_and_bidirectional_characters():
    for value in ("py\nthon", "py\u202ethon", "py\x7fthon"):
        with pytest.raises(ValueError, match="control"):
            _check_name(value)


def test_check_name_preserves_clean_visible_names():
    assert _check_name("python-runtime") == "python-runtime"
