import os
import unittest
from unittest.mock import patch

from settings import env_bool, env_float, env_http_url, env_int, env_path, env_secret, env_text


class EnvironmentParserTests(unittest.TestCase):
    def test_defaults_are_returned_when_variables_are_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_text("TEXT", "fallback"), "fallback")
            self.assertEqual(env_secret("TOKEN"), "")
            self.assertEqual(env_int("COUNT", 3), 3)
            self.assertEqual(env_float("RATE", 0.5), 0.5)
            self.assertTrue(env_bool("ENABLED", True))

    def test_text_rejects_padding_empty_control_and_oversize_values(self):
        invalid = (" padded", "padded ", "", "bad\x00value", "bad\tvalue", "bad\x7fvalue", "123456")
        for value in invalid:
            with self.subTest(value=value), patch.dict(os.environ, {"VALUE": value}, clear=True):
                with self.assertRaises(ValueError):
                    env_text("VALUE", "ok", max_length=5)

    def test_secret_parser_allows_empty_but_rejects_unsafe_header_values(self):
        for value in ("", "token-123", "abc.def_456"):
            with self.subTest(value=value), patch.dict(os.environ, {"TOKEN": value}, clear=True):
                self.assertEqual(env_secret("TOKEN"), value)
        for value in (" padded", "padded ", "line\nbreak", "tab\tvalue", "123456"):
            with self.subTest(value=value), patch.dict(os.environ, {"TOKEN": value}, clear=True):
                with self.assertRaises(ValueError):
                    env_secret("TOKEN", max_length=5)

    def test_integer_parser_enforces_ascii_and_bounds(self):
        for value in ("0", "3", "10", "-2"):
            with self.subTest(value=value), patch.dict(os.environ, {"COUNT": value}, clear=True):
                result = env_int("COUNT", 5, minimum=-2, maximum=10)
                self.assertEqual(result, int(value))
        for value in ("+2", "2.0", "１２", "11", "-3"):
            with self.subTest(value=value), patch.dict(os.environ, {"COUNT": value}, clear=True):
                with self.assertRaises(ValueError):
                    env_int("COUNT", 5, minimum=-2, maximum=10)

    def test_float_parser_rejects_nonfinite_and_out_of_range_values(self):
        with patch.dict(os.environ, {"RATE": "0.75"}, clear=True):
            self.assertEqual(env_float("RATE", 0.5, minimum=0, maximum=1), 0.75)
        for value in ("nan", "inf", "-0.1", "1.1", "wat"):
            with self.subTest(value=value), patch.dict(os.environ, {"RATE": value}, clear=True):
                with self.assertRaises(ValueError):
                    env_float("RATE", 0.5, minimum=0, maximum=1)

    def test_boolean_parser_supports_explicit_common_spellings(self):
        for value, expected in (("1", True), ("true", True), ("YES", True), ("off", False), ("0", False)):
            with self.subTest(value=value), patch.dict(os.environ, {"FLAG": value}, clear=True):
                self.assertIs(env_bool("FLAG", False), expected)
        with patch.dict(os.environ, {"FLAG": "maybe"}, clear=True), self.assertRaises(ValueError):
            env_bool("FLAG", False)

    def test_http_url_parser_requires_clean_service_base_url(self):
        with patch.dict(os.environ, {"URL": "https://example.com/api/"}, clear=True):
            self.assertEqual(env_http_url("URL", "http://localhost"), "https://example.com/api")
        invalid = (
            "file:///tmp/x",
            "https:///missing",
            "https://u:p@example.com",
            "https://example.com/api?token=1",
            "https://example.com/api#section",
        )
        for value in invalid:
            with self.subTest(value=value), patch.dict(os.environ, {"URL": value}, clear=True):
                with self.assertRaises(ValueError):
                    env_http_url("URL", "http://localhost")

    def test_path_parser_expands_home_directory(self):
        with patch.dict(os.environ, {"HOME": "/tmp/test-home", "PATH_VALUE": "~/data.db"}, clear=True):
            self.assertEqual(env_path("PATH_VALUE", "default.db"), "/tmp/test-home/data.db")


if __name__ == "__main__":
    unittest.main()
