import sys
from types import SimpleNamespace

import tools


class _StopEvent:
    def wait(self, timeout=None):
        return False

    def is_set(self):
        return False


def test_camera_frame_error_hides_backend_detail(monkeypatch):
    def fail_frame():
        raise RuntimeError("camera backend path /Users/private/device")

    fake_perception = SimpleNamespace(
        _stop_event=_StopEvent(),
        get_latest_frame=fail_frame,
        query_moondream=lambda frame: "unused",
    )
    monkeypatch.setitem(sys.modules, "perception", fake_perception)

    result = tools.observe_camera.invoke({})

    assert result == "Error: camera frame unavailable"
    assert "/Users/private" not in result
