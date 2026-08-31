from network_validation import valid_hostname


def test_service_dns_labels_are_ascii_only():
    assert valid_hostname("service_name.internal")
    assert not valid_hostname("服务.internal")
    assert not valid_hostname("exämple.internal")
