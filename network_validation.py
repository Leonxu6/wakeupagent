"""Shared network-address validation for browser and service URL boundaries."""
from __future__ import annotations

import ipaddress
import re
import string
import unicodedata
from urllib.parse import unquote

_MAX_DNS_NAME = 253
_MAX_DNS_LABEL = 63
_MAX_ZONE_ID = 64
_HEX_DIGITS = frozenset(string.hexdigits)
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def decode_safe_url_path(path: object, *, field: str = "url") -> str:
    """Decode a URL path only when its percent encoding is unambiguous and safe."""
    if not isinstance(path, str):
        raise ValueError(f"{field} path must be text")
    if _INVALID_PERCENT_ESCAPE.search(path):
        raise ValueError(f"{field} path contains invalid percent encoding")
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} path contains invalid UTF-8 percent encoding") from exc
    if "\\" in decoded or any(
        unicodedata.category(ch) in {"Cc", "Cf", "Cs"} for ch in decoded
    ):
        raise ValueError(f"{field} path contains unsafe encoded characters")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError(f"{field} path must not contain dot segments")
    return decoded


def _valid_ip_literal(hostname: str) -> bool:
    base, marker, zone = hostname.partition("%")
    if marker and (
        not zone
        or zone in {".", ".."}
        or not zone.isascii()
        or len(zone) > _MAX_ZONE_ID
        or not all(ch.isalnum() or ch in {".", "_", "-"} for ch in zone)
    ):
        return False
    try:
        address = ipaddress.ip_address(base)
    except ValueError:
        return False
    # Zone identifiers are an IPv6 scope mechanism. Accept them only where a
    # scope is meaningful; global/loopback addresses with zones are ambiguous
    # across HTTP clients and operating systems.
    if marker and (
        address.version != 6
        or not (address.is_link_local or address.is_multicast)
    ):
        return False
    return True


def _numeric_address_token(label: str) -> bool:
    if label.isascii() and label.isdigit():
        return True
    lower = label.lower()
    return lower.startswith("0x") and len(lower) > 2 and all(ch in _HEX_DIGITS for ch in lower[2:])


def valid_hostname(hostname: object) -> bool:
    """Accept bounded IP literals, localhost, and service/DNS-style hostnames."""
    if not isinstance(hostname, str) or not hostname:
        return False
    if _valid_ip_literal(hostname):
        return True
    if hostname.lower() == "localhost":
        return True
    if not hostname.isascii() or len(hostname) > _MAX_DNS_NAME:
        return False
    if hostname.startswith(".") or hostname.endswith(".") or ".." in hostname:
        return False
    labels = hostname.split(".")
    # libc resolvers accept historical numeric IPv4 spellings such as 127.1,
    # 2130706433, and dotted hex tokens. Reject those ambiguous forms unless
    # ``ipaddress`` already accepted the canonical literal above.
    if labels and all(_numeric_address_token(label) for label in labels):
        return False
    for label in labels:
        if (
            not label
            or len(label) > _MAX_DNS_LABEL
            or label.startswith("-")
            or label.endswith("-")
            or not any(ch.isalnum() for ch in label)
        ):
            return False
        if not all(ch.isalnum() or ch in {"-", "_"} for ch in label):
            return False
    return True
