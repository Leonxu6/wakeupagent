from network_validation import valid_hostname


def test_ipv6_zone_identifiers_are_ascii_only():
    assert valid_hostname("fe80::1%en0")
    assert not valid_hostname("fe80::1%网卡")
