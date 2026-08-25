"""Check GitHub Actions workflows for basic least-privilege and ref hygiene."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from maintenance.common import print_failures, require_root

_ACTION_REF = re.compile(r"\buses:\s*([^\s#]+)")


def audit(root: Path) -> list[str]:
    root = require_root(root)
    workflow_dir = root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return [".github/workflows: workflow directory is missing"]
    workflows = sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")])
    if not workflows:
        return [".github/workflows: no workflow files found"]

    failures: list[str] = []
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root)
        if "pull_request_target:" in text:
            failures.append(f"{rel}: pull_request_target is forbidden for this repository")
        if not re.search(r"(?ms)^permissions:\s*\n\s+contents:\s*read\s*$", text):
            failures.append(f"{rel}: workflow must declare permissions contents: read")
        for ref in _ACTION_REF.findall(text):
            if "@" not in ref:
                failures.append(f"{rel}: action reference lacks an explicit version: {ref}")
                continue
            version = ref.rsplit("@", 1)[1].lower()
            if version in {"main", "master", "head"}:
                failures.append(f"{rel}: action reference must not track a moving branch: {ref}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
