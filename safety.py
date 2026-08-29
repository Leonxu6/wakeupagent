"""Input validation helpers for side-effecting WakeUpAgent actions."""
from __future__ import annotations

import ipaddress
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


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _validator_options(field: object, max_length: object, allow_newlines: object) -> tuple[str, int, bool]:
    if not isinstance(field, str) or not field or field != field.strip() or _has_disallowed_control(field, allow_newlines=False):
        raise ValueError("field must be clean non-empty text")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not isinstance(allow_newlines, bool):
        raise ValueError("allow_newlines must be a boolean")
    return field, max_length, allow_newlines


def require_text(value: object, *, field: str, max_length: int, allow_newlines: bool = False) -> str:
    """Return validated text or raise ``ValueError`` with a stable message."""
    field, max_length, allow_newlines = _validator_options(field, max_length, allow_newlines)
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


def _valid_hostname(hostname: str) -> bool:
    """Accept IP literals/localhost/DNS names while rejecting empty or malformed labels."""
    try:
        ipaddress.ip_address(hostname.split("%", 1)[0])
        return True
    except ValueError:
        pass
    if hostname == "localhost":
        return True
    if hostname.startswith(".") or hostname.endswith(".") or ".." in hostname:
        return False
    labels = hostname.split(".")
    for label in labels:
        if not label or len(label) > 63 or label.startswith("-") or label.endswith("-"):
            return False
        if not all(ch.isalnum() or ch == "-" for ch in label):
            return False
    return True


def require_http_url(value: object, *, max_length: int = 2048) -> str:
    """Validate a browser target without allowing credentials or ambiguous URL syntax."""
    url = require_text(value, field="url", max_length=max_length)
    if any(ch.isspace() for ch in url):
        raise ValueError("url must not contain whitespace")
    if "\\" in url:
        raise ValueError("url must not contain backslashes")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("url is malformed") from exc
    hostname = parsed.hostname
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("url must use http:// or https:// and include a hostname")
    if not _valid_hostname(hostname):
        raise ValueError("url hostname is malformed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url must not contain embedded credentials")
    if parsed.netloc.endswith(":") or port == 0:
        raise ValueError("url must use a valid non-zero port when a port is present")
    return url


def require_app_name(value: object) -> str:
    """Validate an application/process name before passing it to OS tools."""
    app_name = require_text(value, field="app_name", max_length=80)
    if not _APP_NAME.fullmatch(app_name) or not app_name.strip("."):
        raise ValueError("app_name contains unsupported characters")
    return app_name


def require_positive_number(value: object, *, field: str, minimum: float = 0.0, maximum: float | None = None) -> float:
    """Normalize a finite numeric value bounded above an exclusive minimum."""
    minimum_value = _finite_number(minimum, field="minimum")
    maximum_value = None if maximum is None else _finite_number(maximum, field="maximum")
    if maximum_value is not None and maximum_value <= minimum_value:
        raise ValueError("maximum must be greater than minimum")
    number = _finite_number(value, field=field)
    if number <= minimum_value:
        raise ValueError(f"{field} must be greater than {minimum_value:g}")
    if maximum_value is not None and number > maximum_value:
        raise ValueError(f"{field} must be at most {maximum_value:g}")
    return number
