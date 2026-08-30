import os
import unittest
from unittest.mock import patch

from settings import env_http_url


class SettingsHostnameLengthTests(unittest.TestCase):
    def test_service_urls_reject_dns_names_longer_than_protocol_limit(self):
        hostname = ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 62])
        self.assertGreater(len(hostname), 253)
        with patch.dict(os.environ, {"URL": f"https://{hostname}"}, clear=True):
            with self.assertRaisesRegex(ValueError, "hostname"):
                env_http_url("URL", "http://localhost")


if __name__ == "__main__":
    unittest.main()
