"""Audit repository Python sources for syntax errors without importing them."""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

_EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def python_files(root: Path) -> list[Path]:
    """Return repository Python files in deterministic order."""
    if not isinstance(root, Path):
        raise ValueError("root must be a pathlib.Path")
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and not any(part in _EXCLUDED_PARTS for part in path.parts)
    )


def audit_python_sources(root: Path) -> list[str]:
    """Return compact syntax-error descriptions for Python files below *root*."""
    if not root.exists():
        return [f"repository root does not exist: {root}"]
    if not root.is_dir():
        return [f"repository root is not a directory: {root}"]

    failures: list[str] = []
    for path in python_files(root):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            detail = " ".join(str(exc).split())
            failures.append(f"{path.relative_to(root)}: {detail[:500]}")
        except OSError as exc:
            detail = " ".join(str(exc).split())
            failures.append(f"{path.relative_to(root)}: {detail[:500]}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="repository root")
    args = parser.parse_args(argv)
    failures = audit_python_sources(Path(args.root))
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
