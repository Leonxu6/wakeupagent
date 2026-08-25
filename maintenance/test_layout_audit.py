"""Check that tracked test modules follow predictable discovery conventions."""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.common import print_failures, require_root, tracked_files


def audit_paths(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    seen: dict[str, Path] = {}
    for path in paths:
        if not path.parts or path.parts[0] != "tests" or path.suffix != ".py":
            continue
        if path.name != "__init__.py" and not path.name.startswith("test_"):
            failures.append(f"{path}: test module must start with test_")
        key = path.as_posix().casefold()
        previous = seen.get(key)
        if previous is not None and previous != path:
            failures.append(f"test path collision: {previous} <-> {path}")
        else:
            seen[key] = path
    return failures


def audit(root: Path) -> list[str]:
    require_root(root)
    failures = audit_paths(tracked_files(root))
    tests = root / "tests"
    if not tests.is_dir():
        failures.append("tests: test directory is missing")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
