from tools import _bounded_detail


def test_bounded_detail_removes_non_whitespace_controls():
    detail = _bounded_detail("driver\x00failed\x7fnow")
    assert detail == "driver failed now"
    assert "\x00" not in detail
    assert "\x7f" not in detail


def test_bounded_detail_neutralizes_generic_unicode_controls():
    detail = _bounded_detail("driver\u200dfailed\u206anow" + chr(0xD800))
    assert detail == "driver failed now"
    assert "\u200d" not in detail
    assert "\u206a" not in detail
    assert chr(0xD800) not in detail
