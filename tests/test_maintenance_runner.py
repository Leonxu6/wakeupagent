from pathlib import Path

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
