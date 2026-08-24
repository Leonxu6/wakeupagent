from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "maintenance" / "model_asset_audit.py"
spec = spec_from_file_location("model_asset_audit", MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_inspect_asset_accepts_nonempty_task_file(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"model")
    status = module.inspect_asset(path)
    assert status.ok is True
    assert status.detail == "5 bytes"


def test_inspect_asset_does_not_restat_with_is_file(tmp_path, monkeypatch):
    path = tmp_path / "model.task"
    path.write_bytes(b"model")

    def fail_is_file(self):
        raise AssertionError("inspect_asset should rely on the existing stat result")

    monkeypatch.setattr(Path, "is_file", fail_is_file)
    assert module.inspect_asset(path).ok is True


def test_inspect_asset_reports_missing_empty_directory_and_extension(tmp_path):
    assert module.inspect_asset(tmp_path / "missing.task").detail == "missing"
    empty = tmp_path / "empty.task"
    empty.write_bytes(b"")
    assert module.inspect_asset(empty).detail == "empty"
    directory = tmp_path / "dir.task"
    directory.mkdir()
    assert module.inspect_asset(directory).detail == "not a file"
    wrong = tmp_path / "model.bin"
    wrong.write_bytes(b"x")
    assert module.inspect_asset(wrong).detail == "unexpected extension"
