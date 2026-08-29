import sys
from types import SimpleNamespace

import tools


class StopEvent:
    def wait(self, timeout=None):
        return False

    def is_set(self):
        return False


def test_camera_description_is_returned_but_not_logged(monkeypatch):
    private_description = "Private whiteboard project codename"
    perception = SimpleNamespace(
        _stop_event=StopEvent(),
        get_latest_frame=lambda: object(),
        query_moondream=lambda _frame: private_description,
    )
    monkeypatch.setitem(sys.modules, "perception", perception)
    logs = []
    monkeypatch.setattr(tools.console, "print", lambda message: logs.append(str(message)))

    assert tools.observe_camera.invoke({}) == private_description
    assert private_description not in "\n".join(logs)
    assert "description captured" in "\n".join(logs)
