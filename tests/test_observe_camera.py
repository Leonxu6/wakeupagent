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
        generation = 7
        return types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_perception_session_generation=lambda: generation,
            _session_is_current=lambda value: value == generation,
            wait_for_latest_frame=lambda **_kwargs: object(),
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
    def test_rejects_hidden_unicode_camera_descriptions(self, console_print):
        for description in ("person\u200dreading", "person\u206areading", "person" + chr(0xD800) + "reading"):
            with self.subTest(description=repr(description)), patch.dict(
                sys.modules,
                {"perception": self._perception(description)},
            ):
                result = observe_camera.invoke({})
            self.assertEqual(result, "Error: camera description contains control characters")
        rendered = " ".join(str(call.args[0]) for call in console_print.call_args_list if call.args)
        self.assertNotIn("person reading", rendered)

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
            get_perception_session_generation=lambda: 7,
            _session_is_current=lambda value: value == 7,
            wait_for_latest_frame=lambda **_kwargs: object(),
            query_moondream=_boom,
        )
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "Error: camera description failed")
        self.assertNotIn("model backend", result)

    @patch("tools.console.print")
    def test_reports_camera_frame_failures_without_calling_model(self, console_print):
        calls = []

        def _frame_boom(**_kwargs):
            raise RuntimeError("camera backend\nfailed")

        fake = types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_perception_session_generation=lambda: 7,
            _session_is_current=lambda value: value == 7,
            wait_for_latest_frame=_frame_boom,
            query_moondream=lambda frame: calls.append(frame),
        )
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "Error: camera frame unavailable")
        self.assertNotIn("camera backend", result)
        self.assertEqual(calls, [])

    @patch("tools.console.print")
    @patch("tools.time.monotonic", return_value=123.5)
    def test_requires_a_frame_captured_after_observation_started(self, _monotonic, _console_print):
        waits = []
        fake = types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_perception_session_generation=lambda: 7,
            _session_is_current=lambda value: value == 7,
            wait_for_latest_frame=lambda **kwargs: waits.append(kwargs),
            query_moondream=lambda frame: "unused",
        )
        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})
        self.assertEqual(result, "camera not available")
        self.assertEqual(
            waits,
            [{"captured_after": 123.5, "timeout": 30.0}],
        )

    @patch("tools.console.print")
    def test_discards_description_when_perception_restarts_during_inference(self, console_print):
        state = {"generation": 7}

        def restart_during_inference(frame):
            state["generation"] = 8
            return "stale private observation"

        fake = types.SimpleNamespace(
            _stop_event=_StopEvent(),
            get_perception_session_generation=lambda: state["generation"],
            _session_is_current=lambda value: value == state["generation"],
            wait_for_latest_frame=lambda **_kwargs: object(),
            query_moondream=restart_during_inference,
        )

        with patch.dict(sys.modules, {"perception": fake}):
            result = observe_camera.invoke({})

        self.assertEqual(result, "observation cancelled: perception session changed")
        rendered = " ".join(str(call.args[0]) for call in console_print.call_args_list if call.args)
        self.assertNotIn("stale private observation", rendered)


if __name__ == "__main__":
    unittest.main()
