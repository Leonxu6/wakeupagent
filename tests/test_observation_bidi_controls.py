import unittest

from tools import _observation_text


class ObservationBidiControlTests(unittest.TestCase):
    def test_camera_descriptions_reject_directional_controls(self):
        for text in ("reading\u202ebook", "coding\u2066now", "focus\u200fmode"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "control"):
                _observation_text(text)


if __name__ == "__main__":
    unittest.main()
