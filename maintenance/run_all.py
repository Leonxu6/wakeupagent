"""Run the repository's maintenance audits as one deterministic local/CI command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_AUDITS = (
    "ci_contract_audit.py",
    "env_example_audit.py",
    "python_version_audit.py",
    "model_asset_audit.py",
    "python_compile_audit.py",
    "text_integrity_audit.py",
    "repository_hygiene_audit.py",
    "docs_contract_audit.py",
    "env_parity_audit.py",
    "gitignore_contract_audit.py",
    "workflow_security_audit.py",
    "side_effect_contract_audit.py",
    "cli_contract_audit.py",
    "pyproject_metadata_audit.py",
    "dependency_lock_audit.py",
    "readme_command_audit.py",
    "path_case_audit.py",
    "secret_filename_audit.py",
    "test_layout_audit.py",
    "source_import_audit.py",
)


def _validate_scripts(scripts: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(scripts, tuple):
        raise ValueError("scripts must be a tuple of audit filenames")
    seen: set[str] = set()
    for script in scripts:
        if (
            not isinstance(script, str)
            or not script
            or script != script.strip()
            or not script.endswith(".py")
            or "/" in script
            or "\\" in script
        ):
            raise ValueError("audit script names must be simple .py filenames")
        if script in seen:
            raise ValueError("audit script names must be unique")
        seen.add(script)
    return scripts


def _failure_detail(result: object) -> str:
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    detail = " ".join(f"{stdout}\n{stderr}".split()) or "audit failed"
    return detail[:1000]


def run_audits(root: Path, *, scripts: tuple[str, ...] = _AUDITS) -> list[str]:
    if not isinstance(root, Path) or not root.is_dir():
        raise ValueError("root must be an existing directory")
    scripts = _validate_scripts(scripts)
    failures: list[str] = []
    maintenance = root / "maintenance"
    for script in scripts:
        path = maintenance / script
        if not path.is_file():
            failures.append(f"{script}: audit script is missing")
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(path), str(root)],
                cwd=root,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{script}: audit timed out")
            continue
        except OSError as exc:
            failures.append(f"{script}: audit could not start ({exc})")
            continue
        if result.returncode:
            failures.append(f"{script}: {_failure_detail(result)}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    failures = run_audits(Path(args.root))
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
