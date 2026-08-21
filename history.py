"""Small bounded context buffer shared by perception and graph orchestration."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


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
        if not isinstance(text, str):
            return
        self._summary = text.strip()[: self.summary_limit]

    def add_observation(self, text: object) -> None:
        if isinstance(text, str) and text.strip():
            self._items.append(f"[Obs] {text.strip()[: self.observation_limit]}")

    def add_decision(self, text: object) -> None:
        if isinstance(text, str) and text.strip():
            self._items.append(f"[Brain] {text.strip()[: self.decision_limit]}")

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
