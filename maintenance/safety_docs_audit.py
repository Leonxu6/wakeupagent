"""Keep documented side-effect boundaries aligned with the runtime's opt-in model."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root

_REQUIRED = (
    "WAKEUP_ALLOW_TTS",
    "WAKEUP_ALLOW_BROWSER_CONTROL",
    "WAKEUP_ALLOW_EXTERNAL_MESSAGING",
    "WAKEUP_ALLOW_PROCESS_CONTROL",
)


def audit(root: Path) -> list[str]:
    root = require_root(root)
    path = root / "docs" / "side-effects.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"docs/side-effects.md: could not read safety documentation ({exc})"]
    failures = [f"docs/side-effects.md: missing opt-in flag {flag}" for flag in _REQUIRED if flag not in text]
    lowered = text.lower()
    if "chaos" not in lowered or "not registered" not in lowered:
        failures.append("docs/side-effects.md: must state that legacy chaos behavior is not registered")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
