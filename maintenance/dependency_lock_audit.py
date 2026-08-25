"""Verify the checked-in uv lockfile is present and aligned with Python metadata."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from maintenance.common import print_failures, require_root


def audit(root: Path) -> list[str]:
    root = require_root(root)
    try:
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        lock = (root / "uv.lock").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"dependency lock: could not read project metadata or uv.lock ({exc})"]
    failures: list[str] = []
    project_match = re.search(r'^requires-python\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    lock_match = re.search(r'^requires-python\s*=\s*"([^"]+)"', lock, re.MULTILINE)
    if not project_match or not lock_match:
        failures.append("dependency lock: requires-python metadata is missing")
    elif project_match.group(1) != lock_match.group(1):
        failures.append("dependency lock: uv.lock Python requirement does not match pyproject.toml")
    if len(lock) < 1000 or "[[package]]" not in lock:
        failures.append("dependency lock: uv.lock is unexpectedly incomplete")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
