from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "maintenance" / "env_example_audit.py"
spec = spec_from_file_location("env_example_audit", MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_audit_accepts_comments_blank_lines_and_clean_settings(tmp_path):
    path = tmp_path / ".env.example"
    path.write_text("# comment\n\nFOO=bar\nEMPTY=\nDEEPSEEK_API_KEY=\n", encoding="utf-8")
    assert module.audit_env_example(path) == []


def test_audit_reports_malformed_duplicate_and_padded_settings(tmp_path):
    path = tmp_path / ".env.example"
    path.write_text("bad-key=x\nFOO=one\nFOO=two\nBAR= padded \nMISSING\n", encoding="utf-8")
    messages = [issue.message for issue in module.audit_env_example(path)]
    assert "invalid environment variable name" in messages
    assert "duplicate setting: FOO" in messages
    assert "BAR value has surrounding whitespace" in messages
    assert "setting must contain '='" in messages


def test_audit_rejects_populated_secret_like_settings(tmp_path):
    path = tmp_path / ".env.example"
    path.write_text(
        "SERVICE_API_KEY=real-looking-value\n"
        "SESSION_TOKEN=token\n"
        "DB_PASSWORD=password\n"
        "WEBHOOK_SECRET=secret\n",
        encoding="utf-8",
    )
    messages = [issue.message for issue in module.audit_env_example(path)]
    assert messages == [
        "SERVICE_API_KEY must be empty in the tracked template",
        "SESSION_TOKEN must be empty in the tracked template",
        "DB_PASSWORD must be empty in the tracked template",
        "WEBHOOK_SECRET must be empty in the tracked template",
    ]
