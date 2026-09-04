import pytest

from text_safety import model_text, single_line_text


def test_single_line_text_removes_controls_and_bidi_markers():
    assert single_line_text("  hello\n\u202eworld\t  ", limit=50) == "hello world"


def test_single_line_text_rejects_invalid_limits():
    for value in (0, -1, True, 100_001):
        with pytest.raises(ValueError):
            single_line_text("x", limit=value)


def test_model_text_preserves_structured_text_blocks():
    content = ["one", {"type": "text", "text": "two\nlines"}, {"type": "image", "url": "ignored"}]
    assert model_text(content, limit=50, block_limit=10) == "one two lines"


def test_model_text_stops_at_character_budget_without_overconsuming_blocks():
    class GuardedList(list):
        def __getitem__(self, item):
            if isinstance(item, slice):
                return super().__getitem__(item)
            return super().__getitem__(item)

    content = GuardedList([{"text": "a" * 8}, {"text": "b" * 8}, {"text": "c" * 8}])
    assert model_text(content, limit=12, block_limit=3) == "aaaaaaaa bbb"


def test_model_text_rejects_invalid_block_limit():
    with pytest.raises(ValueError):
        model_text("x", block_limit=0)
