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


def test_report_persistence_check_rejects_symlink_target(tmp_path):
    target = tmp_path / "private.md"
    target.write_text("existing", encoding="utf-8")
    link = tmp_path / "daily.md"
    link.symlink_to(target)

    check = _persistence_parent_check(
        "report-dir",
        link,
        allow_missing_parent=True,
        reject_symlink_target=True,
    )

    assert not check.ok
    assert "symbolic link" in check.detail


def test_persistence_check_validates_symlink_policy_type(tmp_path):
    check = _persistence_parent_check(
        "report-dir",
        tmp_path / "daily.md",
        reject_symlink_target="yes",
    )

    assert not check.ok
    assert "must be boolean" in check.detail
