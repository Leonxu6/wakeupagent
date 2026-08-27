from pathlib import Path

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
