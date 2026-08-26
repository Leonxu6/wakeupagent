from main import _message_text


def test_message_text_replaces_control_characters():
    text = _message_text("hello\x00world\x7fagain")
    assert text == "hello world again"
    assert "\x00" not in text
    assert "\x7f" not in text
