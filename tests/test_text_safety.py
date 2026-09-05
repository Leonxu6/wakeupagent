import pytest

from text_safety import model_text, single_line_text


def test_single_line_text_removes_controls_and_bidi_markers():
    assert single_line_text("  hello\n\u202eworld\t  ", limit=50) == "hello world"


def test_single_line_text_removes_hidden_format_and_c1_controls():
    value = "a\u0085b\u200bc\u2060d\ufeffe"
    assert single_line_text(value, limit=50) == "a b c d e"


def test_single_line_text_replaces_unencodable_surrogates():
    assert single_line_text("a" + chr(0xD800) + "b", limit=50) == "a b"


def test_single_line_text_rejects_invalid_limits():
    for value in (0, -1, True, 100_001):
        with pytest.raises(ValueError):
            single_line_text("x", limit=value)


def test_model_text_preserves_structured_text_blocks():
    content = ["one", {"type": "text", "text": "two\nlines"}, {"type": "image", "url": "ignored"}]
    assert model_text(content, limit=50, block_limit=10) == "one two lines"


def test_model_text_ignores_non_text_blocks_even_when_they_have_text_fields():
    content = [
        {"type": "image", "text": "image metadata must not leak"},
        {"type": "tool_result", "text": "tool payload must not leak"},
        {"type": "output_text", "text": "safe answer"},
        {"text": "legacy text"},
    ]
    assert model_text(content, limit=100, block_limit=10) == "safe answer legacy text"


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
