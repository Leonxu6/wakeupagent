"""Check the repository's minimal CI contract using only the workflow text."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CiContract:
    ok: bool
    missing: tuple[str, ...]


def audit_ci_contract(path: Path) -> CiContract:
    text = path.read_text(encoding="utf-8")
    required = {
        "main push trigger": ("push:",),
        "pull request trigger": ("pull_request:",),
        "Python provisioning": ("actions/setup-python@", "astral-sh/setup-uv@"),
        "locked dependency sync": ("uv sync --frozen",),
        "test execution": ("pytest",),
    }
    missing = tuple(
        name for name, alternatives in required.items()
        if not any(needle in text for needle in alternatives)
    )
    return CiContract(not missing, missing)


def main() -> int:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    result = audit_ci_contract(workflow)
    if result.missing:
        print("Missing CI contract elements: " + ", ".join(result.missing))
    else:
        print("CI contract OK")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
