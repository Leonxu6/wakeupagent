"""Input validation helpers for side-effecting WakeUpAgent actions.

These helpers keep validation deterministic and testable without importing the
macOS automation stack. Side-effecting tools should validate inputs before
opening browsers, spawning subprocesses, touching contacts, or calling models.
"""
from __future__ import annotations

import math
import re
from urllib.parse import urlparse

_APP_NAME = re.compile(r"^[\w .+()\-]{1,80}$", re.UNICODE)


def _has_disallowed_control(value: str, *, allow_newlines: bool) -> bool:
    for ch in value:
        code = ord(ch)
        if ch in "\n\r" and allow_newlines:
            continue
        if code < 32 or code == 127:
            return True
    return False


def require_text(
    value: object,
    *,
    field: str,
    max_length: int,
    allow_newlines: bool = False,
) -> str:
    """Return validated text or raise ``ValueError`` with a stable message."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    if not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    if _has_disallowed_control(value, allow_newlines=allow_newlines):
        raise ValueError(f"{field} contains control characters")
    if not allow_newlines and ("\n" in value or "\r" in value):
        raise ValueError(f"{field} must be a single line")
    return value


def require_http_url(value: object, *, max_length: int = 2048) -> str:
    """Validate a browser target without allowing credentials or ambiguous URL syntax."""
    url = require_text(value, field="url", max_length=max_length)
    if any(ch.isspace() for ch in url):
        raise ValueError("url must not contain whitespace")
    if "\\" in url:
        raise ValueError("url must not contain backslashes")
    try:
        parsed = urlparse(url)
        _ = parsed.port  # force malformed port validation
    except (TypeError, ValueError) as exc:
        raise ValueError("url is malformed") from exc
    hostname = parsed.hostname
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("url must use http:// or https:// and include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url must not contain embedded credentials")
    return url


def require_app_name(value: object) -> str:
    """Validate an application/process name before passing it to OS tools."""
    app_name = require_text(value, field="app_name", max_length=80)
    if not _APP_NAME.fullmatch(app_name) or not app_name.strip("."):
        raise ValueError("app_name contains unsupported characters")
    return app_name


def require_positive_number(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    """Normalize a finite positive numeric configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= minimum:
        raise ValueError(f"{field} must be greater than {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum:g}")
    return number
