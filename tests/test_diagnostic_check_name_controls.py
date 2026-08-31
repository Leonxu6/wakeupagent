import pytest

from diagnostics import _check_name


def test_check_name_sanitizes_line_controls_but_rejects_bidi_controls():
    assert _check_name("py\nthon") == "py thon"
    assert _check_name("py\x7fthon") == "py thon"
    with pytest.raises(ValueError, match="bidirectional"):
        _check_name("py\u202ethon")


def test_check_name_preserves_clean_visible_names():
    assert _check_name("python-runtime") == "python-runtime"
