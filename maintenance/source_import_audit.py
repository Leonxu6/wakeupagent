"""Parse core runtime modules and reject relative imports in the flat module layout."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from maintenance.common import print_failures, require_root

_SOURCE_FILES = (
    "config.py", "diagnostics.py", "graph.py", "history.py", "main.py",
    "perception.py", "safety.py", "settings.py", "tools.py",
)


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel in _SOURCE_FILES:
        path = root / rel
        if not path.is_file():
            failures.append(f"{rel}: core runtime module is missing")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append(f"{rel}: could not parse core runtime module ({exc})")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                failures.append(f"{rel}:{node.lineno}: relative imports are unsupported in the flat runtime layout")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
