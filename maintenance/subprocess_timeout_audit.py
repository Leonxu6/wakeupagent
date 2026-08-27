"""Detect subprocess helper calls that can wait forever without a timeout."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files

_HELPERS = {"run", "call", "check_call", "check_output"}


def audit_source(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "subprocess" or node.func.attr not in _HELPERS:
            continue
        if not any(keyword.arg == "timeout" for keyword in node.keywords):
            failures.append(f"subprocess.{node.func.attr}() without timeout on line {node.lineno}")
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
