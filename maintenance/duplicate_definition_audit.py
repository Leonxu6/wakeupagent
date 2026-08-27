"""Detect duplicate function, class, and method definitions in the same scope."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files


def _duplicates(body: list[ast.stmt], *, scope: str) -> list[str]:
    seen: set[str] = set()
    failures: list[str] = []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name in seen:
            failures.append(f"duplicate definition {scope}{node.name} on line {node.lineno}")
        seen.add(node.name)
    return failures


def audit_source(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    failures = _duplicates(tree.body, scope="")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            failures.extend(_duplicates(node.body, scope=f"{node.name}."))
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
