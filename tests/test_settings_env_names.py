import os
import unittest
from unittest.mock import patch

from settings import env_bool, env_float, env_int, env_json_string_map, env_secret, env_text


class EnvironmentNameTests(unittest.TestCase):
    def test_accepts_shell_safe_names_across_parsers(self):
        parsers = (
            lambda name: env_text(name, "ok"),
            lambda name: env_secret(name),
            lambda name: env_int(name, 1),
            lambda name: env_float(name, 1.0),
            lambda name: env_bool(name, False),
            lambda name: env_json_string_map(name, {}),
        )
        with patch.dict(os.environ, {}, clear=True):
            for name in ("WAKEUP_X1", "_PRIVATE", "A"):
                for parser in parsers:
                    with self.subTest(name=name, parser=parser):
                        parser(name)

    def test_rejects_names_that_shell_env_files_cannot_portably_export(self):
        parsers = (
            lambda name: env_text(name, "ok"),
            lambda name: env_secret(name),
            lambda name: env_int(name, 1),
            lambda name: env_float(name, 1.0),
            lambda name: env_bool(name, False),
            lambda name: env_json_string_map(name, {}),
        )
        with patch.dict(os.environ, {}, clear=True):
            for name in ("9BAD", "BAD.NAME", "BAD-KEY", "变量"):
                for parser in parsers:
                    with self.subTest(name=name, parser=parser), self.assertRaises(ValueError):
                        parser(name)


if __name__ == "__main__":
    unittest.main()
