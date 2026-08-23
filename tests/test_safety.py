import math
import unittest

from safety import require_app_name, require_http_url, require_positive_number, require_text


class RequireTextTests(unittest.TestCase):
    def test_accepts_clean_text(self):
        self.assertEqual(require_text("focus", field="message", max_length=20), "focus")

    def test_rejects_non_strings_padding_empty_and_controls(self):
        for value in (None, 7, "", " focus", "focus ", "bad\x00text", "bad\ttext", "two\nlines"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_text(value, field="message", max_length=20)

    def test_allows_newlines_when_explicitly_requested_but_not_other_controls(self):
        self.assertEqual(
            require_text("line one\nline two", field="message", max_length=30, allow_newlines=True),
            "line one\nline two",
        )
        with self.assertRaises(ValueError):
            require_text("line one\tline two", field="message", max_length=30, allow_newlines=True)

    def test_enforces_length_limit(self):
        with self.assertRaisesRegex(ValueError, "at most 4"):
            require_text("12345", field="message", max_length=4)


class UrlValidationTests(unittest.TestCase):
    def test_accepts_http_and_https_urls(self):
        for url in ("http://example.com", "https://example.com/path?q=1"):
            with self.subTest(url=url):
                self.assertEqual(require_http_url(url), url)

    def test_rejects_non_http_hostless_credentials_bad_ports_and_ambiguous_syntax(self):
        invalid = (
            "file:///tmp/x",
            "https:///missing-host",
            "https://user:pass@example.com/private",
            "https://example.com:not-a-port",
            "https://exa mple.com/private",
            "https://example.com/a b",
            "https://example.com\\@evil.test/path",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                require_http_url(url)


class AppNameValidationTests(unittest.TestCase):
    def test_accepts_common_application_names(self):
        for name in ("Safari", "Visual Studio Code", "Steam Helper", "App-2.0"):
            with self.subTest(name=name):
                self.assertEqual(require_app_name(name), name)

    def test_rejects_shell_applescript_metacharacters_and_dot_only_names(self):
        for name in ('Safari" to quit', "Steam; rm -rf ~", "$(whoami)", "../Safari", ".", "..", "..."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                require_app_name(name)


class PositiveNumberValidationTests(unittest.TestCase):
    def test_normalizes_valid_numbers(self):
        self.assertEqual(require_positive_number(5, field="timeout", maximum=10), 5.0)
        self.assertEqual(require_positive_number(0.5, field="timeout", maximum=10), 0.5)

    def test_rejects_bool_nonfinite_nonpositive_and_above_maximum(self):
        for value in (True, False, "5", math.nan, math.inf, -1, 0, 11):
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_positive_number(value, field="timeout", maximum=10)

    def test_rejects_invalid_nonfinite_and_reversed_bounds(self):
        cases = (
            (True, 10),
            (0, False),
            ("0", 10),
            (0, "10"),
            (math.nan, 10),
            (0, math.inf),
            (10, 10),
            (11, 10),
        )
        for minimum, maximum in cases:
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                require_positive_number(5, field="timeout", minimum=minimum, maximum=maximum)  # type: ignore[arg-type]

    def test_normalizes_huge_integer_overflow_to_value_errors(self):
        huge = 10**10000
        for value, minimum, maximum in ((huge, 0, None), (1, -huge, 10), (1, 0, huge)):
            with self.subTest(value=value, minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                require_positive_number(value, field="timeout", minimum=minimum, maximum=maximum)


if __name__ == "__main__":
    unittest.main()
