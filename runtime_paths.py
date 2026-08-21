"""Helpers for resolving WakeUpAgent runtime files independent of cwd."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_runtime_path(value: str | Path) -> Path:
    """Resolve a configured runtime path relative to the repository root."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("runtime path must not be empty")
        if "\x00" in value:
            raise ValueError("runtime path must not contain null bytes")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()
