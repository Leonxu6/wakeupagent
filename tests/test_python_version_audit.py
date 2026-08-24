from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "maintenance" / "python_version_audit.py"
spec = spec_from_file_location("python_version_audit", MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_parse_version_accepts_minor_and_patch_versions():
    assert module.parse_version("3.12") == (3, 12)
    assert module.parse_version("3.12.7\n") == (3, 12)


def test_audit_detects_matching_and_mismatched_python_floors(tmp_path):
    (tmp_path / ".python-version").write_text("3.12\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.12"\n', encoding="utf-8")
    assert module.audit_python_version(tmp_path).ok is True
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.13"\n', encoding="utf-8")
    result = module.audit_python_version(tmp_path)
    assert result.ok is False
    assert "!= requires-python 3.13" in result.detail


def test_audit_ignores_requires_python_text_outside_project_metadata(tmp_path):
    (tmp_path / ".python-version").write_text("3.12\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '# requires-python = ">=9.99"\n[project]\nname = "demo"\nrequires-python = ">=3.12"\n',
        encoding="utf-8",
    )
    assert module.audit_python_version(tmp_path).ok is True


def test_audit_rejects_non_floor_requires_python_constraints(tmp_path):
    (tmp_path / ".python-version").write_text("3.12\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = "~=3.12"\n', encoding="utf-8")
    result = module.audit_python_version(tmp_path)
    assert result.ok is False
    assert "simple >= requires-python floor" in result.detail
