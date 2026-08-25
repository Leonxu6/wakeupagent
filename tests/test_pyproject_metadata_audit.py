from pathlib import Path

from maintenance.pyproject_metadata_audit import audit


def test_pyproject_metadata_accepts_expected_contract(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="wakeupagent"\nreadme="README.md"\nrequires-python=">=3.12"\ndependencies=["rich>=13"]\n', encoding="utf-8")
    assert audit(tmp_path) == []


def test_pyproject_metadata_reports_drift(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="other"\nreadme="README.rst"\nrequires-python=">=3.11"\ndependencies=[]\n', encoding="utf-8")
    failures = audit(tmp_path)
    assert len(failures) == 4
