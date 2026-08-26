"""Validate tracked JSON files without importing runtime code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files


def audit_file(path: Path) -> list[str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid JSON: {path.name}: {exc}"]
    return []


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel in tracked_files(root):
        if rel.suffix.lower() == ".json":
            failures.extend(f"{rel}: {item}" for item in audit_file(root / rel))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    return print_failures(audit(Path(parser.parse_args(argv).root)))


if __name__ == "__main__":
    raise SystemExit(main())
