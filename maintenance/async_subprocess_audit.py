"""Detect synchronous subprocess helpers called directly from async functions."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files

_HELPERS = {"run", "call", "check_call", "check_output", "Popen"}


def _blocking_calls(node: ast.AsyncFunctionDef) -> list[str]:
    failures: list[str] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "subprocess"
            and child.func.attr in _HELPERS
        ):
            failures.append(f"subprocess.{child.func.attr}() inside async function {node.name} on line {child.lineno}")
    return failures


def audit_source(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            failures.extend(_blocking_calls(node))
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
