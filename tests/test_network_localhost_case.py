from network_validation import valid_hostname


def test_localhost_is_case_insensitive_like_dns_names():
    assert valid_hostname("localhost")
    assert valid_hostname("LOCALHOST")
    assert valid_hostname("LocalHost")
