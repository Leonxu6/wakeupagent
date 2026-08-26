from tools import _safe_url_label


def test_safe_url_label_keeps_only_origin():
    assert _safe_url_label("https://example.com/private/token?query=secret#frag") == "https://example.com"


def test_safe_url_label_preserves_explicit_port():
    assert _safe_url_label("http://localhost:11434/private") == "http://localhost:11434"
