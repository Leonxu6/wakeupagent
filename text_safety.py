"""Shared text normalization for model, diagnostic, and orchestration boundaries."""
from __future__ import annotations

import unicodedata

_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}
_HIDDEN_FORMATS = _BIDI_CONTROLS | {"\u00ad", "\u200b", "\u2060", "\ufeff"}
_MAX_TEXT_LIMIT = 100_000
_MAX_BLOCKS = 100


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _safe_visible_char(ch: str) -> str:
    """Replace control/surrogate and selected invisible formatting characters with spaces."""
    if unicodedata.category(ch) in {"Cc", "Cs"} or ch in _HIDDEN_FORMATS:
        return " "
    return ch


def single_line_text(value: object, *, limit: int) -> str:
    """Return bounded visible single-line text, or an empty string for non-text values."""
    limit = _positive_int(limit, field="limit", maximum=_MAX_TEXT_LIMIT)
    if not isinstance(value, str):
        return ""
    clean = "".join(_safe_visible_char(ch) for ch in value)
    return " ".join(clean.split())[:limit].rstrip()


def model_text(content: object, *, limit: int = 2000, block_limit: int = 20) -> str:
    """Extract bounded text from common LangChain/OpenAI structured content shapes."""
    limit = _positive_int(limit, field="limit", maximum=_MAX_TEXT_LIMIT)
    block_limit = _positive_int(block_limit, field="block_limit", maximum=_MAX_BLOCKS)
    if isinstance(content, str):
        return single_line_text(content, limit=limit)
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    used = 0
    for block in content[:block_limit]:
        if isinstance(block, str):
            raw = block
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            raw = block["text"]
        else:
            continue
        remaining = limit - used
        if remaining <= 0:
            break
        text = single_line_text(raw, limit=remaining)
        if not text:
            continue
        separator = 1 if parts else 0
        if separator and used + separator >= limit:
            break
        if separator:
            used += 1
        parts.append(text[: limit - used])
        used += len(parts[-1])
        if used >= limit:
            break
    return " ".join(parts)[:limit].rstrip()
