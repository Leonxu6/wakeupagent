"""Check repository-local Markdown links without making network requests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from maintenance.common import tracked_files

_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_MAX_MARKDOWN_SOURCE_BYTES = 1_048_576


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    target: str


def _markdown_sources(root: Path) -> list[Path]:
    """Return regular tracked Markdown sources without following symlinks."""
    sources: list[Path] = []
    for rel in tracked_files(root):
        if rel.suffix.lower() != ".md":
            continue
        source = root / rel
        if source.is_symlink() or not source.is_file():
            continue
        sources.append(source)
    return sources


def _read_markdown_source(source: Path, *, root: Path) -> str:
    """Read a repository Markdown source with an explicit per-file budget."""
    rel = source.relative_to(root)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ValueError(f"could not inspect Markdown source: {rel}") from exc
    if size > _MAX_MARKDOWN_SOURCE_BYTES:
        raise ValueError(
            f"Markdown source exceeds {_MAX_MARKDOWN_SOURCE_BYTES} byte audit limit: {rel}"
        )
    try:
        return source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read Markdown source as UTF-8: {rel}") from exc


def broken_local_links(root: Path) -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    root_resolved = root.resolve()
    for source in _markdown_sources(root):
        text = _read_markdown_source(source, root=root)
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
    try:
        broken = broken_local_links(root)
    except ValueError as exc:
        print(exc)
        return 1
    for item in broken:
        print(f"{item.source}: broken local link -> {item.target}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
