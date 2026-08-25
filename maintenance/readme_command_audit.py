"""Ensure README setup and diagnostics commands match supported entry points."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root

_REQUIRED = (
    "uv sync",
    "uv run main.py --check",
    "uv run main.py --check-json",
)


def audit(root: Path) -> list[str]:
    root = require_root(root)
    try:
        text = (root / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"README.md: could not read setup documentation ({exc})"]
    return [f"README.md: missing supported command `{command}`" for command in _REQUIRED if command not in text]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
