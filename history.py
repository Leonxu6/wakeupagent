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
    without_controls = "".join(ch if ord(ch) >= 32 and ord(ch) != 127 else " " for ch in value)
    normalized = " ".join(without_controls.split())
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

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-friendly copy suitable for diagnostics or controlled persistence."""
        return {
            "version": 1,
            "limits": {
                "max_items": self.max_items,
                "summary_limit": self.summary_limit,
                "observation_limit": self.observation_limit,
                "decision_limit": self.decision_limit,
            },
            "summary": self._summary,
            "items": list(self._items),
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "ContextHistory":
        """Restore a snapshot while re-validating every external value and bound."""
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("history snapshot must be a version 1 object")
        allowed_fields = {"version", "limits", "summary", "items"}
        if set(snapshot) != allowed_fields:
            raise ValueError("history snapshot is incomplete or contains unknown fields")
        limits = snapshot.get("limits")
        if not isinstance(limits, dict):
            raise ValueError("history snapshot limits must be an object")
        allowed_limits = {"max_items", "summary_limit", "observation_limit", "decision_limit"}
        if set(limits) != allowed_limits:
            raise ValueError("history snapshot limits are incomplete or contain unknown fields")
        history = cls(**{name: limits[name] for name in allowed_limits})

        summary = snapshot["summary"]
        if not isinstance(summary, str):
            raise ValueError("history snapshot summary must be text")
        normalized_summary = _bounded_text(summary, limit=history.summary_limit)
        if normalized_summary != summary:
            raise ValueError("history snapshot summary is not normalized or exceeds its limit")
        history._summary = summary

        items = snapshot["items"]
        if not isinstance(items, list):
            raise ValueError("history snapshot items must be a list")
        if len(items) > history.max_items:
            raise ValueError("history snapshot exceeds max_items")
        previous: str | None = None
        for item in items:
            if not isinstance(item, str):
                raise ValueError("history snapshot items must be text")
            if not (item.startswith("[Obs] ") or item.startswith("[Brain] ")):
                raise ValueError("history snapshot item has an unknown entry type")
            limit = history.observation_limit if item.startswith("[Obs] ") else history.decision_limit
            prefix = "[Obs] " if item.startswith("[Obs] ") else "[Brain] "
            payload = item[len(prefix):]
            if not payload or _bounded_text(payload, limit=limit) != payload:
                raise ValueError("history snapshot item is not normalized or exceeds its limit")
            if item == previous:
                raise ValueError("history snapshot contains consecutive duplicate items")
            history._items.append(item)
            previous = item
        return history

    def __len__(self) -> int:
        return len(self._items)
