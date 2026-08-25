from pathlib import Path

from maintenance.cli_contract_audit import audit


def test_cli_contract_accepts_expected_modes(tmp_path: Path):
    (tmp_path / "main.py").write_text(
        'mode = parser.add_mutually_exclusive_group()\n'
        'mode.add_argument("--graph")\n'
        'mode.add_argument("--check")\n'
        'mode.add_argument("--check-json")\n'
        'return diagnostics_exit_code(checks)\n',
        encoding="utf-8",
    )
    assert audit(tmp_path) == []


def test_cli_contract_reports_missing_safe_diagnostics(tmp_path: Path):
    (tmp_path / "main.py").write_text('mode.add_argument("--graph")\n', encoding="utf-8")
    failures = audit(tmp_path)
    assert any("--check-json" in item for item in failures)
    assert any("diagnostic exit code" in item for item in failures)
