import unittest

from settings import env_bool, env_float, env_int, env_json_string_map, env_secret, env_text


class SettingsEnvironmentNameLengthTests(unittest.TestCase):
    def test_parsers_reject_excessively_long_environment_names(self):
        name = "A" * 129
        parsers = (
            lambda: env_text(name, "ok"),
            lambda: env_secret(name),
            lambda: env_int(name, 1),
            lambda: env_float(name, 1.0),
            lambda: env_bool(name, False),
            lambda: env_json_string_map(name, {}),
        )
        for parser in parsers:
            with self.subTest(parser=parser), self.assertRaisesRegex(ValueError, "at most 128"):
                parser()


if __name__ == "__main__":
    unittest.main()
