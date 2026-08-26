"""Validate tracked TOML files with Python's standard parser."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files


def audit_file(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"invalid TOML: {path.name}: {exc}"]
    return []


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel in tracked_files(root):
        if rel.suffix.lower() == ".toml":
            failures.extend(f"{rel}: {item}" for item in audit_file(root / rel))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    return print_failures(audit(Path(parser.parse_args(argv).root)))


if __name__ == "__main__":
    raise SystemExit(main())
