"""Helpers for resolving WakeUpAgent runtime files independent of cwd."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_runtime_path(value: str | Path) -> Path:
    """Resolve a configured runtime path relative to the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()
