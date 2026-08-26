"""Reject direct URL/VCS project dependencies that bypass normal index review."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from maintenance.common import print_failures, require_root


def audit_dependencies(dependencies: object) -> list[str]:
    if not isinstance(dependencies, list):
        return ["project.dependencies must be a list"]
    failures: list[str] = []
    for item in dependencies:
        if not isinstance(item, str):
            failures.append("project dependency entries must be strings")
            continue
        lowered = item.lower()
        if " @ " in item or lowered.startswith(("git+", "file:", "http://", "https://")):
            failures.append(f"direct dependency source is not allowed: {item}")
    return failures


def audit(root: Path) -> list[str]:
    root = require_root(root)
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"could not parse pyproject.toml: {exc}"]
    project = data.get("project")
    if not isinstance(project, dict):
        return ["pyproject.toml is missing [project]"]
    return audit_dependencies(project.get("dependencies", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    return print_failures(audit(Path(parser.parse_args(argv).root)))


if __name__ == "__main__":
    raise SystemExit(main())
