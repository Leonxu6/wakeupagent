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
    def test_rejects_multiline_camera_descriptions_before_logging(self, console_print):
        fake = self._perception("person\n  reading   notes")
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "Error: camera description contains control characters")
        rendered = " ".join(str(call.args[0]) for call in console_print.call_args_list if call.args)
        self.assertNotIn("reading notes", rendered)

    @patch("tools.console.print")
    def test_rejects_non_text_or_empty_camera_descriptions(self, console_print):
        for description in (None, "   "):
            with self.subTest(description=description), patch.dict(sys.modules, {"perception": self._perception(description)}):
                result = observe_camera.invoke({})
            self.assertIn("Error", result)
            self.assertIn("camera description", result)

    @patch("tools.console.print")
    def test_reports_camera_model_failures_without_raising_from_tool(self, console_print):
        def _boom(frame):
            raise RuntimeError("model backend\nfailed")

        fake = types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_latest_frame=lambda: object(),
            query_moondream=_boom,
        )
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "Error: camera description failed")
        self.assertNotIn("model backend", result)

    @patch("tools.console.print")
    def test_reports_camera_frame_failures_without_calling_model(self, console_print):
        calls = []

        def _frame_boom():
            raise RuntimeError("camera backend\nfailed")

        fake = types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_latest_frame=_frame_boom,
            query_moondream=lambda frame: calls.append(frame),
        )
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "Error: camera frame unavailable")
        self.assertNotIn("camera backend", result)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
