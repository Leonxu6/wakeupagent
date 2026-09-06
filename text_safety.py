"""Shared text normalization for model, diagnostic, and orchestration boundaries."""
from __future__ import annotations

import unicodedata

_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}
_HIDDEN_FORMATS = _BIDI_CONTROLS | {"\u00ad", "\u200b", "\u2060", "\ufeff"}
_TEXT_BLOCK_TYPES = {None, "text", "input_text", "output_text"}
_MAX_TEXT_LIMIT = 100_000
_MAX_RAW_TEXT_CHARS = 200_000
_MAX_BLOCKS = 100


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _safe_visible_char(ch: str) -> str:
    """Replace controls, surrogates, and invisible format characters with spaces."""
    if unicodedata.category(ch) in {"Cc", "Cf", "Cs"} or ch in _HIDDEN_FORMATS:
        return " "
    return ch


def single_line_text(value: object, *, limit: int) -> str:
    """Return bounded visible single-line text, or an empty string for non-text values.

    Raw input inspection is capped as well as output size so a tiny requested result cannot
    force normalization to walk an arbitrarily large model or environment string.
    """
    limit = _positive_int(limit, field="limit", maximum=_MAX_TEXT_LIMIT)
    if not isinstance(value, str):
        return ""
    raw = value[:_MAX_RAW_TEXT_CHARS]
    clean = "".join(_safe_visible_char(ch) for ch in raw)
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
    for index, block in enumerate(content):
        if index >= block_limit:
            break
        if isinstance(block, str):
            raw = block
        elif (
            isinstance(block, dict)
            and block.get("type") in _TEXT_BLOCK_TYPES
            and isinstance(block.get("text"), str)
        ):
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
