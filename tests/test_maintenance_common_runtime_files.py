from pathlib import Path

import pytest

import maintenance.common as common


def test_production_python_files_excludes_tests_and_maintenance(monkeypatch, tmp_path):
    monkeypatch.setattr(
        common,
        "tracked_files",
        lambda root: [
            Path("config.py"),
            Path("graph.py"),
            Path("tests/test_config.py"),
            Path("maintenance/common.py"),
            Path("README.md"),
        ],
    )

    assert common.production_python_files(tmp_path) == [Path("config.py"), Path("graph.py")]


def test_tracked_files_rejects_repository_escape_paths(monkeypatch, tmp_path):
    class Result:
        stdout = b"config.py\0../escape.py\0"

    monkeypatch.setattr(common.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(ValueError, match="outside repository"):
        common.tracked_files(tmp_path)
