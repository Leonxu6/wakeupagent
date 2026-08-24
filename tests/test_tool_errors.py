import unittest

from tools import _bounded_detail


class ToolErrorDetailTests(unittest.TestCase):
    def test_compacts_multiline_driver_output(self):
        self.assertEqual(_bounded_detail(" first\nsecond\tthird "), "first second third")

    def test_bounds_large_error_payloads(self):
        result = _bounded_detail("x" * 1000, limit=80)
        self.assertEqual(result, "x" * 80)

    def test_empty_details_have_stable_fallback(self):
        self.assertEqual(_bounded_detail(" \n\t "), "unknown error")

    def test_broken_string_conversion_uses_type_name(self):
        class Broken:
            def __str__(self):
                raise RuntimeError("render failed")

        self.assertEqual(_bounded_detail(Broken()), "Broken")


if __name__ == "__main__":
    unittest.main()
