"""Detect unbalanced fenced code blocks in tracked Markdown files."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files


def audit_text(text: str) -> list[str]:
    fence: str | None = None
    opening_line = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else None
        if marker is None:
            continue
        if fence is None:
            fence = marker
            opening_line = line_number
        elif marker == fence:
            fence = None
            opening_line = 0
    return [f"unclosed Markdown fence opened on line {opening_line}"] if fence else []


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel in tracked_files(root):
        if rel.suffix.lower() != ".md":
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{rel}: could not read Markdown ({exc})")
            continue
        failures.extend(f"{rel}: {item}" for item in audit_text(text))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    return print_failures(audit(Path(parser.parse_args(argv).root)))


if __name__ == "__main__":
    raise SystemExit(main())
