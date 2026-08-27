"""Detect machine-specific absolute user-home literals in runtime Python code."""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files

_WINDOWS_USER = re.compile(r"^[A-Za-z]:[\\/]Users[\\/]")
_IGNORED_TOP_LEVEL = {"maintenance", "tests"}


def audit_source(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if value.startswith(("/Users/", "/home/")) or _WINDOWS_USER.match(value):
            failures.append(f"machine-specific user path on line {node.lineno}")
    return failures


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel in tracked_files(root):
        if rel.suffix != ".py" or (rel.parts and rel.parts[0] in _IGNORED_TOP_LEVEL):
            continue
        try:
            source = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        failures.extend(f"{rel}: {item}" for item in audit_source(source))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    return print_failures(audit(Path(parser.parse_args(argv).root)))


if __name__ == "__main__":
    raise SystemExit(main())
