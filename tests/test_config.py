import os
import unittest
from unittest.mock import patch

from config import _env_flag


class EnvironmentFlagTests(unittest.TestCase):
    def test_uses_default_when_variable_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_env_flag("MISSING"))
            self.assertTrue(_env_flag("MISSING", default=True))

    def test_accepts_common_true_values_case_insensitively(self):
        for value in ("1", "true", "TRUE", " yes ", "On"):
            with self.subTest(value=value), patch.dict(os.environ, {"FLAG": value}):
                self.assertTrue(_env_flag("FLAG"))

    def test_accepts_common_false_values_case_insensitively(self):
        for value in ("0", "false", "FALSE", " no ", "Off"):
            with self.subTest(value=value), patch.dict(os.environ, {"FLAG": value}):
                self.assertFalse(_env_flag("FLAG", default=True))

    def test_rejects_ambiguous_values(self):
        for value in ("", "enabled", "2", "maybe"):
            with self.subTest(value=value), patch.dict(os.environ, {"FLAG": value}):
                with self.assertRaisesRegex(ValueError, "FLAG must be one of"):
                    _env_flag("FLAG")


if __name__ == "__main__":
    unittest.main()
