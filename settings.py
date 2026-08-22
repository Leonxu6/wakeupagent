"""Typed environment-variable parsing for WakeUpAgent configuration."""
from __future__ import annotations

import math
import os
from pathlib import Path
from urllib.parse import urlparse

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _positive_limit(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _integer_bound(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _float_bound(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def env_text(name: str, default: str, *, max_length: int = 500) -> str:
    max_length = _positive_limit(max_length, field="max_length")
    value = _raw(name)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ValueError(f"{name} default must be a string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} contains control characters")
    return value


def env_secret(name: str, default: str = "", *, max_length: int = 4096) -> str:
    """Read an optional secret while rejecting whitespace and header-breaking controls."""
    max_length = _positive_limit(max_length, field="max_length")
    value = os.getenv(name)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ValueError(f"{name} default must be a string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} contains control characters")
    return value


def env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    minimum = _integer_bound(minimum, field="minimum")
    maximum = _integer_bound(maximum, field="maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    value = _raw(name)
    if value is None:
        if isinstance(default, bool) or not isinstance(default, int):
            raise ValueError(f"{name} default must be an integer")
        result = default
    else:
        if value.startswith("+") or not value.lstrip("-").isascii() or not value.lstrip("-").isdigit():
            raise ValueError(f"{name} must be an integer")
        result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    minimum = _float_bound(minimum, field="minimum")
    maximum = _float_bound(maximum, field="maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    value = _raw(name)
    if value is None:
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            raise ValueError(f"{name} default must be a number")
        result = float(default)
    else:
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return result


def env_bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name} must be one of: 1/0, true/false, yes/no, on/off")


def env_http_url(name: str, default: str) -> str:
    """Read a clean HTTP(S) service base URL without credentials or URL state."""
    value = env_text(name, default, max_length=2048)
    try:
        parsed = urlparse(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid URL") from exc
    hostname = parsed.hostname
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError(f"{name} must be an http(s) URL with a hostname")
    if any(ch.isspace() for ch in hostname):
        raise ValueError(f"{name} hostname must not contain whitespace")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain a query string or fragment")
    return value.rstrip("/")


def env_path(name: str, default: str) -> str:
    value = env_text(name, default, max_length=4096)
    return str(Path(value).expanduser())
