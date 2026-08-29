from tools import _observation_text


def test_observation_truncation_does_not_leave_trailing_whitespace():
    assert _observation_text("ab cd", limit=4) == "ab"


def test_observation_truncation_preserves_clean_text():
    assert _observation_text("abcdef", limit=4) == "abcd"
