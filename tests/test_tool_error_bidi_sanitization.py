import unittest

from tools import _bounded_detail


class ToolErrorBidiSanitizationTests(unittest.TestCase):
    def test_driver_error_details_neutralize_directional_controls(self):
        self.assertEqual(_bounded_detail("before\u202eafter"), "before after")
        self.assertEqual(_bounded_detail("left\u2066right"), "left right")


if __name__ == "__main__":
    unittest.main()
