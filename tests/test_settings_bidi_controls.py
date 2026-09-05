import os
import unittest
from unittest.mock import patch

from settings import env_json_string_map, env_secret, env_text


class SettingsBidiControlTests(unittest.TestCase):
    def test_text_and_secret_values_reject_directional_controls(self):
        for parser in (
            lambda: env_text("VALUE", "safe"),
            lambda: env_secret("VALUE"),
        ):
            with self.subTest(parser=parser), patch.dict(os.environ, {"VALUE": "safe\u202eevil"}, clear=True):
                with self.assertRaisesRegex(ValueError, "control"):
                    parser()

    def test_text_and_secret_values_reject_generic_format_controls(self):
        for hidden in ("\u200c", "\u200d", "\u206a", "\u206f"):
            for parser in (
                lambda: env_text("VALUE", "safe"),
                lambda: env_secret("VALUE"),
            ):
                with self.subTest(hidden=hidden, parser=parser), patch.dict(
                    os.environ, {"VALUE": f"safe{hidden}evil"}, clear=True
                ):
                    with self.assertRaisesRegex(ValueError, "control"):
                        parser()

    def test_json_map_aliases_and_values_reject_directional_controls(self):
        for raw in ('{"mentor\\u202e":"Alice"}', '{"mentor":"Ali\\u2066ce"}'):
            with self.subTest(raw=raw), patch.dict(os.environ, {"CONTACTS": raw}, clear=True):
                with self.assertRaisesRegex(ValueError, "control"):
                    env_json_string_map("CONTACTS", {})

    def test_json_map_aliases_and_values_reject_generic_format_controls(self):
        for raw in ('{"mentor\\u200d":"Alice"}', '{"mentor":"Ali\\u206ace"}'):
            with self.subTest(raw=raw), patch.dict(os.environ, {"CONTACTS": raw}, clear=True):
                with self.assertRaisesRegex(ValueError, "control"):
                    env_json_string_map("CONTACTS", {})


if __name__ == "__main__":
    unittest.main()
