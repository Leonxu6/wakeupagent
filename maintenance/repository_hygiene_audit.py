"""Reject generated or private artifacts that have accidentally become tracked."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files

_FORBIDDEN_NAMES = {".DS_Store", "Thumbs.db", ".env", "superego.db", "superego.db-shm", "superego.db-wal"}
_FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "venv"}
_FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def audit_paths(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for rel in paths:
        if rel.name in _FORBIDDEN_NAMES:
            failures.append(f"{rel}: private/generated artifact is tracked")
        elif any(part in _FORBIDDEN_PARTS for part in rel.parts):
            failures.append(f"{rel}: generated directory content is tracked")
        elif rel.suffix.lower() in _FORBIDDEN_SUFFIXES:
            failures.append(f"{rel}: compiled Python artifact is tracked")
    return failures


def audit(root: Path) -> list[str]:
    require_root(root)
    return audit_paths(tracked_files(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
