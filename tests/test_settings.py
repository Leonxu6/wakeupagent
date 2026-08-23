import os
import unittest
from unittest.mock import patch

from settings import env_bool, env_float, env_http_url, env_int, env_json_string_map, env_path, env_secret, env_text


class EnvironmentParserTests(unittest.TestCase):
    def test_defaults_are_returned_when_variables_are_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_text("TEXT", "fallback"), "fallback")
            self.assertEqual(env_secret("TOKEN"), "")
            self.assertEqual(env_int("COUNT", 3), 3)
            self.assertEqual(env_float("RATE", 0.5), 0.5)
            self.assertTrue(env_bool("ENABLED", True))

    def test_text_defaults_are_validated_when_variable_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_text("TEXT", "fallback", max_length=8), "fallback")
            for default in (" padded", "padded ", "", "bad\nvalue", "toolong"):
                with self.subTest(default=default), self.assertRaises(ValueError):
                    env_text("TEXT", default, max_length=6)

    def test_secret_defaults_are_validated_when_variable_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_secret("TOKEN", "safe-token"), "safe-token")
            for default in (" padded", "padded ", "line\nbreak", "123456"):
                with self.subTest(default=default), self.assertRaises(ValueError):
                    env_secret("TOKEN", default, max_length=5)

    def test_secret_default_must_be_text(self):
        with patch.dict(os.environ, {}, clear=True):
            for default in (None, 7, False, ["token"]):
                with self.subTest(default=default), self.assertRaises(ValueError):
                    env_secret("TOKEN", default)  # type: ignore[arg-type]

    def test_text_length_limits_must_be_positive_integers(self):
        with patch.dict(os.environ, {}, clear=True):
            for limit in (0, -1, True, 2.5, "5"):
                with self.subTest(limit=limit):
                    with self.assertRaises(ValueError):
                        env_text("TEXT", "ok", max_length=limit)  # type: ignore[arg-type]
                    with self.assertRaises(ValueError):
                        env_secret("TOKEN", "", max_length=limit)  # type: ignore[arg-type]

    def test_integer_defaults_must_be_real_integers(self):
        with patch.dict(os.environ, {}, clear=True):
            for default in (True, False, 2.5, "3", None):
                with self.subTest(default=default), self.assertRaises(ValueError):
                    env_int("COUNT", default)  # type: ignore[arg-type]

    def test_integer_bounds_must_be_ordered_integers(self):
        with patch.dict(os.environ, {}, clear=True):
            for minimum, maximum in ((True, 10), (0, False), (0.5, 10), (0, "10"), (11, 10)):
                with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                    env_int("COUNT", 5, minimum=minimum, maximum=maximum)  # type: ignore[arg-type]

    def test_float_defaults_must_be_numeric_and_finite(self):
        with patch.dict(os.environ, {}, clear=True):
            for default in (True, False, "0.5", None, float("nan"), float("inf")):
                with self.subTest(default=default), self.assertRaises(ValueError):
                    env_float("RATE", default)  # type: ignore[arg-type]

    def test_float_bounds_must_be_finite_numbers_in_order(self):
        with patch.dict(os.environ, {}, clear=True):
            cases = (
                (True, 1.0),
                (0.0, False),
                ("0", 1.0),
                (0.0, "1"),
                (float("nan"), 1.0),
                (0.0, float("inf")),
                (2.0, 1.0),
            )
            for minimum, maximum in cases:
                with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                    env_float("RATE", 0.5, minimum=minimum, maximum=maximum)  # type: ignore[arg-type]

    def test_boolean_defaults_must_be_booleans(self):
        with patch.dict(os.environ, {}, clear=True):
            for default in (0, 1, "true", None):
                with self.subTest(default=default), self.assertRaises(ValueError):
                    env_bool("FLAG", default)  # type: ignore[arg-type]

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
        for value in ("+2", "--1", "---2", "2.0", "１２", "11", "-3"):
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
            "https://exa mple.com/api",
            "https://example.com/api?token=1",
            "https://example.com/api#section",
            "https://example.com/a b",
            "https://example.com\\@evil.test/path",
        )
        for value in invalid:
            with self.subTest(value=value), patch.dict(os.environ, {"URL": value}, clear=True):
                with self.assertRaises(ValueError):
                    env_http_url("URL", "http://localhost")

    def test_path_parser_expands_home_directory(self):
        with patch.dict(os.environ, {"HOME": "/tmp/test-home", "PATH_VALUE": "~/data.db"}, clear=True):
            self.assertEqual(env_path("PATH_VALUE", "default.db"), "/tmp/test-home/data.db")

    def test_path_parser_normalizes_home_expansion_failures(self):
        with patch.dict(os.environ, {"PATH_VALUE": "~ghost/data.db"}, clear=True):
            with patch("settings.Path.expanduser", side_effect=RuntimeError("unknown home")):
                with self.assertRaisesRegex(ValueError, "could not expand"):
                    env_path("PATH_VALUE", "default.db")

    def test_json_string_map_rejects_duplicate_keys(self):
        with patch.dict(os.environ, {"CONTACTS": '{"mentor":"Alice","mentor":"Bob"}'}, clear=True):
            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                env_json_string_map("CONTACTS", {})


if __name__ == "__main__":
    unittest.main()
