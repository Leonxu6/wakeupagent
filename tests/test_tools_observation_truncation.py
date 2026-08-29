from tools import _observation_text


def test_observation_truncation_does_not_leave_trailing_whitespace():
    assert _observation_text("abc def", limit=4) == "abc"


def test_observation_truncation_preserves_clean_text():
    assert _observation_text("abcdef", limit=4) == "abcd"
