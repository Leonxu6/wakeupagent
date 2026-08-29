from pathlib import Path

from diagnostics import _persistence_parent_check


def test_persistence_check_rejects_empty_padded_and_directory_only_paths(tmp_path):
    for value in ("", " state.db", "state.db ", "bad\npath", Path("."), Path("/")):
        check = _persistence_parent_check("checkpoint-dir", value)
        assert not check.ok


def test_persistence_check_accepts_file_path_under_writable_directory(tmp_path):
    check = _persistence_parent_check("checkpoint-dir", tmp_path / "state.db")
    assert check.ok
    assert check.detail == str(tmp_path.resolve())
