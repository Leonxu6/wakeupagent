import os
from pathlib import Path

import numpy as np

import perception


def test_moondream_writes_encoded_frame_through_original_descriptor(monkeypatch):
    payload = b"synthetic-jpeg"
    seen = {}

    class Client:
        def generate(self, *, model, prompt, images):
            path = Path(images[0])
            seen["path"] = path
            seen["payload"] = path.read_bytes()
            seen["mode"] = path.stat().st_mode & 0o777
            return type("Response", (), {"response": "person reading"})()

    real_write = os.write
    write_sizes = []

    def short_write(fd, data):
        chunk = bytes(data[:3])
        write_sizes.append(len(chunk))
        return real_write(fd, chunk)

    monkeypatch.setattr(perception.cv2, "resize", lambda frame, size: frame)
    monkeypatch.setattr(
        perception.cv2,
        "imencode",
        lambda extension, frame, options: (True, np.frombuffer(payload, dtype=np.uint8)),
    )
    monkeypatch.setattr(perception.os, "write", short_write)
    monkeypatch.setattr(perception, "_ollama_client", Client())

    result = perception.query_moondream(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result == "person reading"
    assert seen["payload"] == payload
    assert seen["mode"] == 0o600
    assert len(write_sizes) > 1
    assert not seen["path"].exists()


def test_moondream_closes_and_removes_temp_file_when_encoding_fails(monkeypatch):
    created = {}
    real_mkstemp = perception.tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created["fd"] = fd
        created["path"] = Path(path)
        return fd, path

    monkeypatch.setattr(perception.cv2, "resize", lambda frame, size: frame)
    monkeypatch.setattr(perception.cv2, "imencode", lambda *args: (False, None))
    monkeypatch.setattr(perception.tempfile, "mkstemp", recording_mkstemp)

    result = perception.query_moondream(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result == "moondream unavailable"
    assert not created["path"].exists()
    try:
        os.fstat(created["fd"])
    except OSError:
        pass
    else:
        raise AssertionError("temporary frame descriptor remained open")
