import os
import unittest
from unittest.mock import patch

from settings import env_json_string_map


class JsonStringMapTests(unittest.TestCase):
    def test_reads_valid_json_map_and_copies_defaults(self):
        default = {"family": "Mom"}
        with patch.dict(os.environ, {}, clear=True):
            result = env_json_string_map("CONTACTS", default)
        self.assertEqual(result, default)
        self.assertIsNot(result, default)

        with patch.dict(os.environ, {"CONTACTS": '{"mentor":"Dr Xu","team":"Study Group"}'}, clear=True):
            self.assertEqual(
                env_json_string_map("CONTACTS", {}),
                {"mentor": "Dr Xu", "team": "Study Group"},
            )

    def test_rejects_invalid_json_shapes_and_unsafe_entries(self):
        invalid = (
            "[]",
            '"text"',
            '{"ok":1}',
            '{" padded":"value"}',
            '{"ok":" padded"}',
            '{"ok":"bad\\nvalue"}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), patch.dict(os.environ, {"CONTACTS": raw}, clear=True):
                with self.assertRaises(ValueError):
                    env_json_string_map("CONTACTS", {})

    def test_enforces_entry_count_and_option_type(self):
        with patch.dict(os.environ, {"CONTACTS": '{"a":"A","b":"B"}'}, clear=True):
            with self.assertRaises(ValueError):
                env_json_string_map("CONTACTS", {}, max_entries=1)
        with patch.dict(os.environ, {}, clear=True):
            for limit in (0, -1, True, 1.5, "10"):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    env_json_string_map("CONTACTS", {}, max_entries=limit)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
