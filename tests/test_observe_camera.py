import sys
import types
import unittest
from unittest.mock import patch

from tools import observe_camera


class _StopEvent:
    def wait(self, timeout):
        self.timeout = timeout

    def is_set(self):
        return False


class ObserveCameraTests(unittest.TestCase):
    def _perception(self, description):
        return types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_latest_frame=lambda: object(),
            query_moondream=lambda frame: description,
        )

    @patch("tools.console.print")
    def test_normalizes_multiline_camera_descriptions_before_logging(self, console_print):
        fake = self._perception("person\n  reading   notes")
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "person reading notes")
        self.assertIn("person reading notes", console_print.call_args.args[0])

    @patch("tools.console.print")
    def test_rejects_non_text_or_empty_camera_descriptions(self, console_print):
        for description in (None, "   "):
            with self.subTest(description=description), patch.dict(sys.modules, {"perception": self._perception(description)}):
                result = observe_camera.invoke({})
            self.assertIn("Error", result)
            self.assertIn("camera description", result)


if __name__ == "__main__":
    unittest.main()
