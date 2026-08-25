"""Verify privacy-sensitive and generated WakeUpAgent paths stay ignored by Git."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root

_REQUIRED = {
    ".env",
    ".venv",
    "__pycache__/",
    "*.py[oc]",
    "build/",
    "dist/",
    "superego.db",
    "superego.db-shm",
    "superego.db-wal",
    "memory/daily_reports.md",
}


def audit(root: Path) -> list[str]:
    root = require_root(root)
    path = root / ".gitignore"
    try:
        lines = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")}
    except (OSError, UnicodeError) as exc:
        return [f".gitignore: could not read ignore rules ({exc})"]
    return [f".gitignore: missing required privacy/generated rule {item}" for item in sorted(_REQUIRED - lines)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
