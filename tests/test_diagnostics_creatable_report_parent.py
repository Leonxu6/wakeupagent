from pathlib import Path
from unittest.mock import patch

import diagnostics


def test_report_parent_check_accepts_missing_directory_when_ancestor_is_writable(tmp_path: Path):
    report = tmp_path / "memory" / "daily" / "report.md"

    check = diagnostics._persistence_parent_check(
        "report-dir",
        report,
        allow_missing_parent=True,
    )

    assert check.ok is True
    assert check.detail.endswith("(creatable)")
    assert not report.parent.exists()


def test_checkpoint_parent_check_still_requires_existing_directory(tmp_path: Path):
    checkpoint = tmp_path / "state" / "agent.db"

    check = diagnostics._persistence_parent_check("checkpoint-dir", checkpoint)

    assert check.ok is False
    assert "missing" in check.detail


def test_report_parent_check_rejects_unwritable_existing_ancestor(tmp_path: Path):
    report = tmp_path / "memory" / "daily" / "report.md"

    with patch("diagnostics.os.access", return_value=False):
        check = diagnostics._persistence_parent_check(
            "report-dir",
            report,
            allow_missing_parent=True,
        )

    assert check.ok is False
    assert "not writable" in check.detail
