"""Verify the checked-in uv lockfile is present and aligned with Python metadata."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from maintenance.common import print_failures, require_root

_MAX_PYPROJECT_BYTES = 512 * 1024
_MAX_LOCK_BYTES = 16 * 1024 * 1024


def _load_toml(path: Path, *, label: str, max_bytes: int) -> tuple[dict[str, object] | None, str | None]:
    try:
        if path.is_symlink():
            return None, f"dependency lock: {label} must not be a symbolic link"
        if not path.is_file():
            return None, f"dependency lock: {label} must be a regular file"
        if path.stat().st_size > max_bytes:
            return None, f"dependency lock: {label} exceeds {max_bytes} byte audit limit"
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return None, f"dependency lock: could not parse {label} ({exc})"
    if not isinstance(parsed, dict):
        return None, f"dependency lock: {label} did not parse as a TOML document"
    return parsed, None


def audit(root: Path) -> list[str]:
    root = require_root(root)
    pyproject, pyproject_error = _load_toml(
        root / "pyproject.toml", label="pyproject.toml", max_bytes=_MAX_PYPROJECT_BYTES
    )
    lock, lock_error = _load_toml(root / "uv.lock", label="uv.lock", max_bytes=_MAX_LOCK_BYTES)
    failures = [error for error in (pyproject_error, lock_error) if error]
    if pyproject is None or lock is None:
        return failures

    project = pyproject.get("project")
    project_requires = project.get("requires-python") if isinstance(project, dict) else None
    lock_requires = lock.get("requires-python")
    if not isinstance(project_requires, str) or not isinstance(lock_requires, str):
        failures.append("dependency lock: requires-python metadata is missing")
    elif project_requires != lock_requires:
        failures.append("dependency lock: uv.lock Python requirement does not match pyproject.toml")

    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        failures.append("dependency lock: uv.lock package table is unexpectedly incomplete")
    elif not all(isinstance(package, dict) and isinstance(package.get("name"), str) for package in packages):
        failures.append("dependency lock: uv.lock contains a malformed package entry")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
