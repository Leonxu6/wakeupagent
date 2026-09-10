import unittest

import tools
from tools import _bounded_detail


class ToolErrorDetailTests(unittest.TestCase):
    def test_compacts_multiline_driver_output(self):
        self.assertEqual(_bounded_detail(" first\nsecond\tthird "), "first second third")

    def test_bounds_large_error_payloads(self):
        result = _bounded_detail("x" * 1000, limit=80)
        self.assertEqual(result, "x" * 80)

    def test_caps_raw_normalization_work_before_sanitizing(self):
        original = tools._MAX_DETAIL_INPUT_CHARS
        try:
            tools._MAX_DETAIL_INPUT_CHARS = 12
            self.assertEqual(_bounded_detail("x" * 1000, limit=80), "x" * 12)
        finally:
            tools._MAX_DETAIL_INPUT_CHARS = original

    def test_rejects_unbounded_output_limits(self):
        with self.assertRaises(ValueError):
            _bounded_detail("error", limit=tools._MAX_DETAIL_LIMIT + 1)

    def test_empty_details_have_stable_fallback(self):
        self.assertEqual(_bounded_detail(" \n\t "), "unknown error")

    def test_broken_string_conversion_uses_type_name(self):
        class Broken:
            def __str__(self):
                raise RuntimeError("render failed")

        self.assertEqual(_bounded_detail(Broken()), "Broken")

    def test_limit_must_be_a_positive_integer(self):
        for limit in (0, -1, True, 1.5, "20"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                _bounded_detail("error", limit=limit)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
