"""Flag direct eval/exec calls in tracked Python sources."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files

_BLOCKED_CALLS = {"eval", "exec"}


def audit_source(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # syntax is owned by python_compile_audit
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            failures.append(f"dynamic code execution via {node.func.id}() on line {node.lineno}")
    return failures


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel in tracked_files(root):
        if rel.suffix != ".py":
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
