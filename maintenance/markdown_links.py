"""Check repository-local Markdown links without making network requests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    target: str


def broken_local_links(root: Path) -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    root_resolved = root.resolve()
    for source in sorted(root.rglob("*.md")):
        text = source.read_text(encoding="utf-8")
        for raw_target in _LINK.findall(text):
            raw_target = raw_target.strip()
            if not raw_target:
                continue
            parsed = urlsplit(raw_target)
            if parsed.scheme or parsed.netloc:
                continue
            target = unquote(parsed.path)
            if not target:
                continue
            candidate = (root / target.lstrip("/")) if target.startswith("/") else (source.parent / target)
            try:
                resolved = candidate.resolve()
                inside_root = resolved.is_relative_to(root_resolved)
            except (OSError, RuntimeError):
                inside_root = False
                resolved = candidate
            if not inside_root or not resolved.exists():
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
