"""Validate core project metadata needed by contributors and tooling."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from maintenance.common import print_failures, require_root


def audit(root: Path) -> list[str]:
    root = require_root(root)
    path = root / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        return [f"pyproject.toml: could not parse project metadata ({exc})"]
    project = data.get("project")
    if not isinstance(project, dict):
        return ["pyproject.toml: [project] table is missing"]
    failures: list[str] = []
    if project.get("name") != "wakeupagent":
        failures.append("pyproject.toml: project name must remain wakeupagent")
    if project.get("readme") != "README.md":
        failures.append("pyproject.toml: project readme must be README.md")
    if project.get("requires-python") != ">=3.12":
        failures.append("pyproject.toml: requires-python must remain >=3.12")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        failures.append("pyproject.toml: project dependencies must be a non-empty list")
    elif len(dependencies) != len(set(dependencies)):
        failures.append("pyproject.toml: duplicate dependency declarations are not allowed")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
