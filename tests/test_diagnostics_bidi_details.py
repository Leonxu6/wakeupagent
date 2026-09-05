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

    def test_reports_strip_hidden_and_c1_controls_through_shared_boundary(self):
        rendered = format_checks([Check("service", False, "a\u0085b\u200bc\u2060d\ufeffe")])
        self.assertIn("a b c d e", rendered)
        self.assertNotIn("\u200b", rendered)
        self.assertNotIn("\u2060", rendered)

    def test_json_reports_survive_unencodable_surrogates(self):
        payload = json.loads(format_checks_json([Check("service", True, "a" + chr(0xD800) + "b")]))
        self.assertEqual(payload[0]["detail"], "a b")


if __name__ == "__main__":
    unittest.main()
