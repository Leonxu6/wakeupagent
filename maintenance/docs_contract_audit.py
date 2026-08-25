"""Check that maintainer-facing documentation remains present and substantive."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root

_REQUIRED = {
    "README.md": 1000,
    "CONTRIBUTING.md": 400,
    "SECURITY.md": 400,
    "docs/architecture.md": 500,
    "docs/configuration.md": 500,
    "docs/diagnostics.md": 500,
    "docs/privacy.md": 500,
    "docs/security-boundaries.md": 500,
    "docs/testing.md": 500,
    "docs/troubleshooting.md": 500,
}


def audit(root: Path) -> list[str]:
    root = require_root(root)
    failures: list[str] = []
    for rel, minimum_size in _REQUIRED.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"{rel}: required maintainer documentation is missing")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{rel}: could not read documentation ({exc})")
            continue
        if len(text.strip()) < minimum_size:
            failures.append(f"{rel}: documentation is unexpectedly small")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
