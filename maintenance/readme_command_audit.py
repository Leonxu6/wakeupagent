"""Ensure contributor-facing setup and diagnostics commands match supported entry points."""
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
    documents = (root / "README.md", root / "docs" / "diagnostics.md")
    chunks: list[str] = []
    for path in documents:
        if not path.exists():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            return [f"{path.relative_to(root)}: could not read command documentation ({exc})"]
    if not chunks:
        return ["README.md: contributor command documentation is missing"]
    text = "\n".join(chunks)
    return [f"documentation: missing supported command `{command}`" for command in _REQUIRED if command not in text]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
