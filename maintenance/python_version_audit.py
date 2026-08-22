"""Check that repository metadata agrees on the supported Python floor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")


@dataclass(frozen=True)
class VersionAudit:
    ok: bool
    detail: str


def parse_version(text: str) -> tuple[int, int]:
    match = _VERSION.fullmatch(text.strip())
    if not match:
        raise ValueError("version must look like major.minor")
    return int(match.group(1)), int(match.group(2))


def audit_python_version(root: Path) -> VersionAudit:
    version_file = root / ".python-version"
    pyproject = root / "pyproject.toml"
    declared = parse_version(version_file.read_text(encoding="utf-8"))
    project_text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)', project_text)
    if not match:
        return VersionAudit(False, "pyproject.toml has no >= requires-python floor")
    floor = parse_version(match.group(1))
    if declared != floor:
        return VersionAudit(False, f".python-version {declared[0]}.{declared[1]} != requires-python {floor[0]}.{floor[1]}")
    return VersionAudit(True, f"Python {declared[0]}.{declared[1]}")


def main() -> int:
    result = audit_python_version(Path(__file__).resolve().parents[1])
    print(result.detail)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
