"""Check that safe diagnostics and explicit runtime modes stay available from the CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root

_REQUIRED_SNIPPETS = {
    '--graph': 'mode.add_argument("--graph"',
    '--check': 'mode.add_argument("--check"',
    '--check-json': 'mode.add_argument("--check-json"',
    'mutually-exclusive modes': 'add_mutually_exclusive_group()',
    'diagnostic exit code': 'return diagnostics_exit_code(checks)',
}


def audit(root: Path) -> list[str]:
    root = require_root(root)
    try:
        text = (root / "main.py").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"main.py: could not read CLI entry point ({exc})"]
    return [f"main.py: missing CLI contract {label}" for label, snippet in _REQUIRED_SNIPPETS.items() if snippet not in text]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
