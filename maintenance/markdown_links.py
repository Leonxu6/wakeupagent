"""Check repository-local Markdown links without making network requests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import unquote

_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    target: str


def broken_local_links(root: Path) -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    for source in sorted(root.rglob("*.md")):
        text = source.read_text(encoding="utf-8")
        for raw_target in _LINK.findall(text):
            target = raw_target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            target = unquote(target)
            resolved = (root / target.lstrip("/")) if target.startswith("/") else (source.parent / target)
            if not resolved.resolve().exists():
                broken.append(BrokenLink(source.relative_to(root), raw_target))
    return broken


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    broken = broken_local_links(root)
    for item in broken:
        print(f"{item.source}: broken local link -> {item.target}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
