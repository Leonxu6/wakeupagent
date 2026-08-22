from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "maintenance" / "ci_contract_audit.py"
spec = spec_from_file_location("ci_contract_audit", MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_ci_contract_accepts_required_workflow_elements(tmp_path):
    path = tmp_path / "ci.yml"
    path.write_text("push:\npull_request:\nactions/setup-python@v5\nuv sync --frozen\npytest -q\n", encoding="utf-8")
    result = module.audit_ci_contract(path)
    assert result.ok is True
    assert result.missing == ()


def test_ci_contract_reports_missing_test_execution(tmp_path):
    path = tmp_path / "ci.yml"
    path.write_text("push:\npull_request:\nactions/setup-python@v5\nuv sync --frozen\n", encoding="utf-8")
    result = module.audit_ci_contract(path)
    assert result.ok is False
    assert "test execution" in result.missing
