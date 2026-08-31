from network_validation import valid_hostname


def test_dot_only_zone_identifiers_are_rejected():
    assert valid_hostname("fe80::1%.") is False
    assert valid_hostname("fe80::1%..") is False


def test_real_interface_style_zone_names_still_work():
    assert valid_hostname("fe80::1%en0.100") is True
