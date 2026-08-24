"""Small bounded context buffer shared by perception and graph orchestration."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _bounded_text(value: object, *, limit: int) -> str:
    """Normalize model/user text into one bounded context line."""
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    return normalized[:limit]


@dataclass
class ContextHistory:
    """Store bounded human-readable context without growing for the process lifetime."""

    max_items: int = 15
    summary_limit: int = 200
    observation_limit: int = 100
    decision_limit: int = 120
    _items: deque[str] = field(init=False, repr=False)
    _summary: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_items = _positive_int(self.max_items, field_name="max_items")
        self.summary_limit = _positive_int(self.summary_limit, field_name="summary_limit")
        self.observation_limit = _positive_int(self.observation_limit, field_name="observation_limit")
        self.decision_limit = _positive_int(self.decision_limit, field_name="decision_limit")
        self._items = deque(maxlen=self.max_items)

    def set_summary(self, text: object) -> None:
        normalized = _bounded_text(text, limit=self.summary_limit)
        if normalized:
            self._summary = normalized

    def clear_summary(self) -> None:
        """Drop the synthesized summary without disturbing recent raw context."""
        self._summary = ""

    def clear(self) -> None:
        """Reset summary and recent entries while preserving configured limits."""
        self._summary = ""
        self._items.clear()

    def _append_unique(self, entry: str) -> None:
        if self._items and self._items[-1] == entry:
            return
        self._items.append(entry)

    def add_observation(self, text: object) -> None:
        normalized = _bounded_text(text, limit=self.observation_limit)
        if normalized:
            self._append_unique(f"[Obs] {normalized}")

    def add_decision(self, text: object) -> None:
        normalized = _bounded_text(text, limit=self.decision_limit)
        if normalized:
            self._append_unique(f"[Brain] {normalized}")

    def render(self, *, recent: int = 10) -> str:
        recent = _positive_int(recent, field_name="recent")
        parts: list[str] = []
        if self._summary:
            parts.append(f"Summary: {self._summary}")
        items = list(self._items)[-recent:]
        if items:
            parts.append("Recent history:\n" + "\n".join(items))
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return len(self._items)
