from tools import _bounded_detail


def test_bounded_detail_removes_non_whitespace_controls():
    detail = _bounded_detail("driver\x00failed\x7fnow")
    assert detail == "driver failed now"
    assert "\x00" not in detail
    assert "\x7f" not in detail
