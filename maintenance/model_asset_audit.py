"""Validate local model assets without opening them or contacting a network service."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPECTED = ("pose_landmarker_lite.task", "gesture_recognizer.task")


@dataclass(frozen=True)
class AssetStatus:
    name: str
    ok: bool
    detail: str


def inspect_asset(path: Path) -> AssetStatus:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return AssetStatus(path.name, False, "missing")
    except OSError as exc:
        return AssetStatus(path.name, False, f"unreadable: {exc}")
    if not path.is_file():
        return AssetStatus(path.name, False, "not a file")
    if stat.st_size == 0:
        return AssetStatus(path.name, False, "empty")
    if path.suffix != ".task":
        return AssetStatus(path.name, False, "unexpected extension")
    return AssetStatus(path.name, True, f"{stat.st_size} bytes")


def audit_assets(root: Path) -> list[AssetStatus]:
    return [inspect_asset(root / name) for name in EXPECTED]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    statuses = audit_assets(root)
    for item in statuses:
        print(f"{'OK' if item.ok else 'FAIL'} {item.name}: {item.detail}")
    return 1 if any(not item.ok for item in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
