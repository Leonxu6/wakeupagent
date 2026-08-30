import unittest

from safety import require_http_url


class SafetyIPv6ZoneTests(unittest.TestCase):
    def test_accepts_bounded_ipv6_zone_identifiers(self):
        self.assertEqual(
            require_http_url("http://[fe80::1%25en0]:8080/status"),
            "http://[fe80::1%25en0]:8080/status",
        )

    def test_rejects_empty_unsafe_and_oversized_ipv6_zones(self):
        invalid = (
            "http://[fe80::1%]:8080/",
            "http://[fe80::1%bad!zone]:8080/",
            f"http://[fe80::1%{'a' * 65}]:8080/",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                require_http_url(url)


if __name__ == "__main__":
    unittest.main()
