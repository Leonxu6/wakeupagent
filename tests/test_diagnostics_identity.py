from pathlib import Path

import pytest

import diagnostics


def test_diagnostic_root_resolves_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = diagnostics._diagnostic_root("models/..")
    assert root == tmp_path.resolve()
    assert root.is_absolute()


def test_diagnostic_root_normalizes_resolution_failures(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(RuntimeError("loop")))
    with pytest.raises(ValueError, match="could not be resolved"):
        diagnostics._diagnostic_root("~/models")


def test_diagnostic_checks_reject_casefold_identity_collisions():
    checks = [
        diagnostics.Check("Python", True, "ok"),
        diagnostics.Check("python", True, "ok"),
    ]
    with pytest.raises(ValueError, match="duplicate diagnostic check name"):
        diagnostics.format_checks(checks)
    with pytest.raises(ValueError):
        diagnostics.format_checks_json(checks)
    with pytest.raises(ValueError):
        diagnostics.diagnostics_exit_code(checks)


def test_nonconflicting_check_names_keep_existing_rendering_contract():
    checks = [
        diagnostics.Check("python", True, "3.12"),
        diagnostics.Check("platform", False, "Linux"),
    ]
    assert diagnostics.format_checks(checks) == "[OK] python: 3.12\n[WARN] platform: Linux"
