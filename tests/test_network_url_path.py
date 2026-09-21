import pytest

from network_validation import decode_safe_url_path


def test_decode_safe_url_path_accepts_valid_percent_encoded_utf8():
    assert decode_safe_url_path("/notes/%E4%B8%AD") == "/notes/中"


@pytest.mark.parametrize("path", ["/%", "/%2", "/%GG", "/%FF"])
def test_decode_safe_url_path_rejects_malformed_percent_encoding(path):
    with pytest.raises(ValueError, match="percent encoding"):
        decode_safe_url_path(path)


@pytest.mark.parametrize("path", ["/safe/%5Csecret", "/safe/%0Aheader"])
def test_decode_safe_url_path_rejects_encoded_controls_and_backslashes(path):
    with pytest.raises(ValueError, match="unsafe encoded characters"):
        decode_safe_url_path(path)


@pytest.mark.parametrize("path", ["/safe/%2e%2e/secret", "/safe/./item"])
def test_decode_safe_url_path_rejects_dot_segments_after_decoding(path):
    with pytest.raises(ValueError, match="dot segments"):
        decode_safe_url_path(path)
