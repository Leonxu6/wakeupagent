import json
import unittest

from diagnostics import Check, format_checks, format_checks_json


class DiagnosticBidiDetailTests(unittest.TestCase):
    def test_text_reports_neutralize_directional_controls(self):
        rendered = format_checks([Check("service", False, "before\u202eafter")])
        self.assertNotIn("\u202e", rendered)
        self.assertIn("before after", rendered)

    def test_json_reports_neutralize_directional_controls(self):
        payload = json.loads(format_checks_json([Check("service", True, "left\u2066right")]))
        self.assertNotIn("\u2066", payload[0]["detail"])
        self.assertEqual(payload[0]["detail"], "left right")


if __name__ == "__main__":
    unittest.main()
