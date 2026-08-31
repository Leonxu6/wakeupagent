"""Typed environment-variable parsing for WakeUpAgent configuration."""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

from network_validation import valid_hostname

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FLOAT_TEXT = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", re.ASCII)
_MAX_ENV_NAME = 128
_MAX_NUMERIC_TEXT = 128
_MAX_MAP_ENTRIES = 1000
_MAX_TEXT_LIMIT = 100_000
_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}


def _env_name(name: object) -> str:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("environment variable name must be clean non-empty text")
    if len(name) > _MAX_ENV_NAME:
        raise ValueError(f"environment variable name must be at most {_MAX_ENV_NAME} characters")
    if not _ENV_NAME.fullmatch(name):
        raise ValueError("environment variable name must use shell-safe letters, digits, and underscores")
    return name


def _raw(name: str) -> str | None:
    name = _env_name(name)
    value = os.getenv(name)
    if value is None:
        return None
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _numeric_raw(name: str) -> str | None:
    value = _raw(name)
    if value is not None and len(value) > _MAX_NUMERIC_TEXT:
        raise ValueError(f"{name} numeric text must be at most {_MAX_NUMERIC_TEXT} characters")
    return value


def _positive_limit(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _text_limit(value: object, *, field: str = "max_length") -> int:
    result = _positive_limit(value, field=field)
    if result > _MAX_TEXT_LIMIT:
        raise ValueError(f"{field} must be at most {_MAX_TEXT_LIMIT}")
    return result


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
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_text(value: object, *, field: str, max_length: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    if not value and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 or ch in _BIDI_CONTROLS for ch in value):
        raise ValueError(f"{field} contains control characters")
    return value


def env_text(name: str, default: str, *, max_length: int = 500) -> str:
    max_length = _text_limit(max_length)
    value = _raw(name)
    if value is None:
        value = default
    return _validate_text(value, field=name, max_length=max_length)


def env_secret(name: str, default: str = "", *, max_length: int = 4096) -> str:
    """Read an optional secret while rejecting whitespace and header-breaking controls."""
    name = _env_name(name)
    max_length = _text_limit(max_length)
    value = os.getenv(name)
    if value is None:
        value = default
    return _validate_text(value, field=name, max_length=max_length, allow_empty=True)


def env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    minimum = _integer_bound(minimum, field="minimum")
    maximum = _integer_bound(maximum, field="maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    value = _numeric_raw(name)
    if value is None:
        if isinstance(default, bool) or not isinstance(default, int):
            raise ValueError(f"{name} default must be an integer")
        result = default
    else:
        digits = value[1:] if value.startswith("-") else value
        if value.startswith("+") or not digits or not digits.isascii() or not digits.isdigit():
            raise ValueError(f"{name} must be an integer")
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
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
    value = _numeric_raw(name)
    if value is None:
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            raise ValueError(f"{name} default must be a number")
        try:
            result = float(default)
        except OverflowError as exc:
            raise ValueError(f"{name} default must be finite") from exc
    else:
        if not _FLOAT_TEXT.fullmatch(value):
            raise ValueError(f"{name} must be a number")
        try:
            result = float(value)
        except (OverflowError, ValueError) as exc:
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
        if not isinstance(default, bool):
            raise ValueError(f"{name} default must be a boolean")
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
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{name} must not contain whitespace")
    if "\\" in value:
        raise ValueError(f"{name} must not contain backslashes")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid URL") from exc
    hostname = parsed.hostname
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError(f"{name} must be an http(s) URL with a hostname")
    if not valid_hostname(hostname):
        raise ValueError(f"{name} hostname is malformed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain a query string or fragment")
    if parsed.netloc.endswith(":") or port == 0:
        raise ValueError(f"{name} must use a valid non-zero port when a port is present")
    return value.rstrip("/")


def env_path(name: str, default: str) -> str:
    value = env_text(name, default, max_length=4096)
    try:
        return str(Path(value).expanduser())
    except RuntimeError as exc:
        raise ValueError(f"{name} could not expand the user home directory") from exc


def env_json_string_map(name: str, default: dict[str, str], *, max_entries: int = 100) -> dict[str, str]:
    """Read a small JSON object whose keys and values are clean, bounded strings."""
    name = _env_name(name)
    max_entries = _positive_limit(max_entries, field="max_entries")
    if max_entries > _MAX_MAP_ENTRIES:
        raise ValueError(f"max_entries must be at most {_MAX_MAP_ENTRIES}")
    raw = os.getenv(name)
    if raw is None:
        value: object = default
    else:
        if not raw or raw != raw.strip():
            raise ValueError(f"{name} must be non-empty JSON without surrounding whitespace")
        if len(raw) > 16384:
            raise ValueError(f"{name} must be at most 16384 characters")

        def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"{name} must not contain duplicate keys")
                result[key] = item
            return result

        try:
            value = json.loads(raw, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if len(value) > max_entries:
        raise ValueError(f"{name} must contain at most {max_entries} entries")
    normalized: dict[str, str] = {}
    aliases: set[str] = set()
    for key, item in value.items():
        clean_key = _validate_text(key, field=f"{name} key", max_length=80)
        clean_value = _validate_text(item, field=f"{name} value", max_length=200)
        alias = unicodedata.normalize("NFKC", clean_key).casefold()
        if alias in aliases:
            raise ValueError(f"{name} must not contain normalization-equivalent duplicate keys")
        aliases.add(alias)
        normalized[clean_key] = clean_value
    return normalized
