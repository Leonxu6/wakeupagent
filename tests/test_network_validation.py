import unittest

from network_validation import valid_hostname


class NetworkValidationTests(unittest.TestCase):
    def test_accepts_local_dns_service_and_ip_hosts(self):
        for host in (
            "localhost",
            "example.com",
            "ollama_service",
            "127.0.0.1",
            "::1",
            "fe80::1%25en0",
            "ff02::1%25en0",
        ):
            with self.subTest(host=host):
                self.assertTrue(valid_hostname(host))

    def test_rejects_malformed_and_oversized_hosts(self):
        invalid = (
            None,
            "",
            ".example.com",
            "example.com.",
            "example..com",
            "-example.com",
            "example-.com",
            "bad!host",
            "a" * 64 + ".com",
            ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 62]),
            "fe80::1%",
            "fe80::1%bad!zone",
            f"fe80::1%{'a' * 65}",
            "127.0.0.1%eth0",
            "192.168.1.5%25en0",
            "::1%25en0",
            "2001:db8::1%25en0",
        )
        for host in invalid:
            with self.subTest(host=host):
                self.assertFalse(valid_hostname(host))

    def test_rejects_noncanonical_numeric_address_spellings(self):
        for host in (
            "127.1",
            "127.0.1",
            "2130706433",
            "0177.0.0.1",
            "0x7f.0x0.0x0.0x1",
            "0x7f000001",
        ):
            with self.subTest(host=host):
                self.assertFalse(valid_hostname(host))


if __name__ == "__main__":
    unittest.main()
