import unittest

import tools
from tools import _observation_text


class ObservationTextTests(unittest.TestCase):
    def test_normalizes_and_bounds_single_line_camera_text(self):
        self.assertEqual(_observation_text(" person   reading ", limit=8), "person r")

    def test_rejects_multiline_camera_text(self):
        with self.assertRaisesRegex(ValueError, "control characters"):
            _observation_text("person\nreading")

    def test_caps_raw_camera_text_before_normalization(self):
        original = tools._MAX_OBSERVATION_INPUT_CHARS
        try:
            tools._MAX_OBSERVATION_INPUT_CHARS = 12
            self.assertEqual(_observation_text("person reading forever", limit=100), "person readi")
        finally:
            tools._MAX_OBSERVATION_INPUT_CHARS = original

    def test_rejects_unbounded_observation_limits(self):
        with self.assertRaises(ValueError):
            _observation_text("person reading", limit=tools._MAX_OBSERVATION_LIMIT + 1)

    def test_limit_must_be_a_positive_integer(self):
        for limit in (0, -1, True, 1.5, "20"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                _observation_text("person reading", limit=limit)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
