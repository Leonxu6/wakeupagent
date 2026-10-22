"""Perception sessions own their camera handle and stop signal."""

import pytest
import numpy as np

import perception


class _ModelPath:
    def exists(self):
        return True


class _Capture:
    def __init__(self, *, opened):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True


def _install_runtime_stubs(monkeypatch, capture):
    monkeypatch.setattr(perception, "_POSE_MODEL", _ModelPath())
    monkeypatch.setattr(perception, "_GESTURE_MODEL", _ModelPath())
    monkeypatch.setattr(perception.cv2, "VideoCapture", lambda index: capture)
    destroyed = []
    monkeypatch.setattr(perception.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    return destroyed


def test_failed_camera_open_releases_handle_and_signals_stop(monkeypatch):
    capture = _Capture(opened=False)
    destroyed = _install_runtime_stubs(monkeypatch, capture)
    perception._stop_event.set()

    perception.run_perception_loop()

    assert capture.released is True
    assert destroyed == [True]
    assert perception._stop_event.is_set()
    assert perception.get_latest_frame() is None


def test_session_clears_stale_stop_and_cleans_up_after_failure(monkeypatch):
    capture = _Capture(opened=True)
    destroyed = _install_runtime_stubs(monkeypatch, capture)
    perception._stop_event.set()

    def fail_after_open(cap, **kwargs):
        assert cap is capture
        assert not perception._stop_event.is_set()
        raise RuntimeError("detector startup failed")

    monkeypatch.setattr(perception, "_run_open_camera", fail_after_open)
    with pytest.raises(RuntimeError, match="detector startup failed"):
        perception.run_perception_loop()

    assert capture.released is True
    assert destroyed == [True]
    assert perception._stop_event.is_set()
    assert perception.get_latest_frame() is None


def test_session_discards_stale_frame_before_model_validation(monkeypatch):
    class MissingModel:
        def exists(self):
            return False

    monkeypatch.setattr(perception, "_POSE_MODEL", MissingModel())
    monkeypatch.setattr(perception, "_latest_raw_frame", object())

    perception.run_perception_loop()

    assert perception.get_latest_frame() is None
    assert perception._stop_event.is_set()


def test_stopped_session_suppresses_late_analysis_callback(monkeypatch):
    delivered = []
    callback = lambda *args: delivered.append(args)
    perception._stop_event.set()

    assert perception._deliver_state_callback(callback, "reading", "12:00:00", True, False) is False
    assert delivered == []


def test_active_session_delivers_analysis_callback(monkeypatch):
    delivered = []
    callback = lambda *args: delivered.append(args)
    perception._stop_event.clear()

    assert perception._deliver_state_callback(callback, "reading", "12:00:00", True, False) is True
    assert delivered == [("reading", "12:00:00", True, False)]


def test_restarted_session_rejects_late_callback_from_previous_generation(monkeypatch):
    delivered = []
    callback = lambda *args: delivered.append(args)
    perception._stop_event.clear()
    previous = perception._begin_perception_session()
    current = perception._begin_perception_session()

    assert perception._deliver_state_callback(
        callback,
        "stale observation",
        "12:00:00",
        False,
        True,
        session_generation=previous,
    ) is False
    assert perception._deliver_state_callback(
        callback,
        "current observation",
        "12:00:01",
        True,
        False,
        session_generation=current,
    ) is True
    assert delivered == [("current observation", "12:00:01", True, False)]


def test_stale_session_cleanup_does_not_stop_or_clear_current_session(monkeypatch):
    capture = _Capture(opened=True)
    destroyed = _install_runtime_stubs(monkeypatch, capture)
    current_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def replace_running_session(cap, **kwargs):
        assert cap is capture
        perception._begin_perception_session()
        perception._stop_event.clear()
        perception._store_latest_frame(current_frame)

    monkeypatch.setattr(perception, "_run_open_camera", replace_running_session)

    perception.run_perception_loop()

    assert capture.released is True
    assert destroyed == []
    assert not perception._stop_event.is_set()
    assert np.array_equal(perception.get_latest_frame(), current_frame)


def test_capture_loop_eligibility_is_scoped_to_current_generation():
    perception._stop_event.clear()
    previous = perception._begin_perception_session()
    current = perception._begin_perception_session()

    assert perception._session_should_run(previous) is False
    assert perception._session_should_run(current) is True

    perception._stop_event.set()
    assert perception._session_should_run(current) is False
