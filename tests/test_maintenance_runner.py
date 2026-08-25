from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import maintenance.run_all as runner


def test_maintenance_runner_reports_missing_scripts(tmp_path: Path):
    (tmp_path / "maintenance").mkdir()
    failures = runner.run_audits(tmp_path, scripts=("missing.py",))
    assert failures == ["missing.py: audit script is missing"]


def test_maintenance_runner_executes_successful_audit(tmp_path: Path):
    maintenance = tmp_path / "maintenance"
    maintenance.mkdir()
    (maintenance / "ok.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    assert runner.run_audits(tmp_path, scripts=("ok.py",)) == []


def test_maintenance_runner_reports_failed_audit_output(tmp_path: Path):
    maintenance = tmp_path / "maintenance"
    maintenance.mkdir()
    (maintenance / "bad.py").write_text('print("broken contract")\nraise SystemExit(1)\n', encoding="utf-8")
    failures = runner.run_audits(tmp_path, scripts=("bad.py",))
    assert failures == ["bad.py: broken contract"]


def test_maintenance_runner_preserves_stdout_and_stderr(tmp_path: Path):
    maintenance = tmp_path / "maintenance"
    maintenance.mkdir()
    (maintenance / "bad.py").write_text("# fixture\n", encoding="utf-8")
    result = SimpleNamespace(returncode=1, stdout="audit context\n", stderr="traceback detail\n")
    with patch.object(runner.subprocess, "run", return_value=result):
        failures = runner.run_audits(tmp_path, scripts=("bad.py",))
    assert failures == ["bad.py: audit context traceback detail"]


@pytest.mark.parametrize(
    "scripts",
    [
        ("../outside.py",),
        ("nested/audit.py",),
        ("nested\\audit.py",),
        (" padded.py",),
        ("audit.txt",),
        ("same.py", "same.py"),
    ],
)
def test_maintenance_runner_rejects_unsafe_or_duplicate_selectors(tmp_path: Path, scripts: tuple[str, ...]):
    (tmp_path / "maintenance").mkdir()
    with pytest.raises(ValueError, match="audit script names"):
        runner.run_audits(tmp_path, scripts=scripts)
