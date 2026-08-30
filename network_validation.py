"""Shared network-address validation for browser and service URL boundaries."""
from __future__ import annotations

import ipaddress

_MAX_DNS_NAME = 253
_MAX_DNS_LABEL = 63
_MAX_ZONE_ID = 64


def _valid_ip_literal(hostname: str) -> bool:
    base, marker, zone = hostname.partition("%")
    if marker and (
        not zone
        or len(zone) > _MAX_ZONE_ID
        or not all(ch.isalnum() or ch in {".", "_", "-"} for ch in zone)
    ):
        return False
    try:
        address = ipaddress.ip_address(base)
    except ValueError:
        return False
    # Zone identifiers are an IPv6 scope mechanism. Accepting one on an IPv4
    # literal creates an address spelling that downstream clients interpret
    # inconsistently, so reject it at the shared URL boundary.
    if marker and address.version != 6:
        return False
    return True


def valid_hostname(hostname: object) -> bool:
    """Accept bounded IP literals, localhost, and service/DNS-style hostnames."""
    if not isinstance(hostname, str) or not hostname:
        return False
    if _valid_ip_literal(hostname):
        return True
    if hostname == "localhost":
        return True
    if len(hostname) > _MAX_DNS_NAME:
        return False
    if hostname.startswith(".") or hostname.endswith(".") or ".." in hostname:
        return False
    for label in hostname.split("."):
        if (
            not label
            or len(label) > _MAX_DNS_LABEL
            or label.startswith("-")
            or label.endswith("-")
        ):
            return False
        if not all(ch.isalnum() or ch in {"-", "_"} for ch in label):
            return False
    return True
