from pathlib import Path
from unittest.mock import patch

import diagnostics


def test_model_check_rejects_regular_file_without_read_permission(tmp_path: Path):
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")

    with patch("diagnostics.os.access", return_value=False):
        check = diagnostics._model_check("pose-model", model)

    assert check.ok is False
    assert check.detail == f"not readable: {model}"


def test_model_check_still_accepts_readable_nonempty_file(tmp_path: Path):
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")

    with patch("diagnostics.os.access", return_value=True):
        check = diagnostics._model_check("pose-model", model)

    assert check.ok is True
    assert check.detail == "pose.task (5 bytes)"
