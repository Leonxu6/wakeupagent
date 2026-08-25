"""Ensure environment variables used by runtime code are represented in .env.example."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from maintenance.common import print_failures, require_root

_ENV_CALL = re.compile(r'\benv_(?:bool|float|http_url|int|json_string_map|path|secret|text)\(\s*["\']([A-Z][A-Z0-9_]*)["\']')
_DIRECT_FLAG = re.compile(r'_feature_enabled\(\s*["\']([A-Z][A-Z0-9_]*)["\']')


def runtime_env_names(root: Path) -> set[str]:
    names: set[str] = set()
    for rel in ("config.py", "diagnostics.py", "tools.py"):
        text = (root / rel).read_text(encoding="utf-8")
        names.update(_ENV_CALL.findall(text))
        names.update(_DIRECT_FLAG.findall(text))
    return names


def template_env_names(text: str) -> set[str]:
    names: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0]
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            names.add(key)
    return names


def audit(root: Path) -> list[str]:
    root = require_root(root)
    try:
        template = (root / ".env.example").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f".env.example: could not read template ({exc})"]
    missing = sorted(runtime_env_names(root) - template_env_names(template))
    return [f".env.example: missing runtime variable {name}" for name in missing]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
