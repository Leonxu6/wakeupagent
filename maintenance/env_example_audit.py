"""Audit .env-style templates without loading secrets into the process environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET")


@dataclass(frozen=True)
class EnvIssue:
    line: int
    message: str


def _looks_sensitive(key: str) -> bool:
    return key.endswith(_SECRET_SUFFIXES)


def audit_env_example(path: Path) -> list[EnvIssue]:
    issues: list[EnvIssue] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in raw:
            issues.append(EnvIssue(number, "setting must contain '='"))
            continue
        key, value = raw.split("=", 1)
        if key != key.strip() or not _KEY.fullmatch(key):
            issues.append(EnvIssue(number, "invalid environment variable name"))
            continue
        if key in seen:
            issues.append(EnvIssue(number, f"duplicate setting: {key}"))
        seen.add(key)
        if value != value.strip():
            issues.append(EnvIssue(number, f"{key} value has surrounding whitespace"))
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            issues.append(EnvIssue(number, f"{key} value contains control characters"))
        if _looks_sensitive(key) and value.strip():
            issues.append(EnvIssue(number, f"{key} must be empty in the tracked template"))
    return issues


def main() -> int:
    path = Path(__file__).resolve().parents[1] / ".env.example"
    issues = audit_env_example(path)
    for issue in issues:
        print(f"{path}:{issue.line}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
