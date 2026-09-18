from pathlib import Path

import pytest

import maintenance.common as common


def test_production_python_files_excludes_tests_and_maintenance(monkeypatch, tmp_path):
    (tmp_path / "config.py").write_text("CONFIG = True\n", encoding="utf-8")
    (tmp_path / "graph.py").write_text("GRAPH = True\n", encoding="utf-8")
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


def test_production_python_files_skips_symlinks_and_non_files(monkeypatch, tmp_path):
    regular = tmp_path / "runtime.py"
    regular.write_text("RUNTIME = True\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-runtime.py"
    outside.write_text("OUTSIDE = True\n", encoding="utf-8")
    symlink = tmp_path / "linked.py"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")
    (tmp_path / "directory.py").mkdir()

    monkeypatch.setattr(
        common,
        "tracked_files",
        lambda root: [
            Path("runtime.py"),
            Path("linked.py"),
            Path("directory.py"),
            Path("missing.py"),
        ],
    )

    assert common.production_python_files(tmp_path) == [Path("runtime.py")]


def test_tracked_files_rejects_repository_escape_paths(monkeypatch, tmp_path):
    class Result:
        stdout = b"config.py\0../escape.py\0"

    monkeypatch.setattr(common.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(ValueError, match="outside repository"):
        common.tracked_files(tmp_path)
