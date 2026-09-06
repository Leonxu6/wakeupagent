"""Fast, side-effect-free diagnostics for WakeUpAgent installations."""
from __future__ import annotations

import importlib
import json
import os
import platform
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from safety import require_http_url
from settings import env_bool
from text_safety import single_line_text


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


_CRITICAL_CHECKS = {
    "python",
    "pose-model",
    "gesture-model",
    "configuration",
    "checkpoint-dir",
    "report-dir",
}
_CRITICAL_IDENTITIES = {
    unicodedata.normalize("NFKC", name).casefold() for name in _CRITICAL_CHECKS
}
_REQUIRED_CONFIG_FIELDS = (
    "CHECKPOINT_DB_PATH",
    "DAILY_REPORT_PATH",
    "OLLAMA_HOST",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
)
_DETAIL_LIMIT = 1000
_MAX_CHECKS = 1000
_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}


def _version_pair(version: object) -> tuple[int, int]:
    if version is None:
        return sys.version_info.major, sys.version_info.minor
    if not isinstance(version, tuple) or len(version) != 2:
        raise ValueError("version must be a (major, minor) tuple")
    major, minor = version
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (major, minor)):
        raise ValueError("version components must be non-negative integers")
    return major, minor


def _python_check(version: tuple[int, int] | None = None) -> Check:
    current = _version_pair(version)
    ok = current >= (3, 12)
    detail = platform.python_version() if version is None else f"{current[0]}.{current[1]}"
    if not ok:
        detail += " (requires Python >=3.12)"
    return Check("python", ok, detail)


def _model_check(name: str, path: Path) -> Check:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return Check(name, False, f"missing: {path}")
    except OSError as exc:
        return Check(name, False, f"unreadable: {path} ({exc})")
    if not stat.S_ISREG(metadata.st_mode):
        return Check(name, False, f"not a file: {path}")
    if metadata.st_size == 0:
        return Check(name, False, f"empty model file: {path}")
    return Check(name, True, f"{path.name} ({metadata.st_size} bytes)")


def _directory_check(name: str, path: Path) -> Check:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return Check(name, False, f"missing: {path}")
    except OSError as exc:
        return Check(name, False, f"unreadable: {path} ({exc})")
    if not stat.S_ISDIR(metadata.st_mode):
        return Check(name, False, f"not a directory: {path}")
    if not os.access(path, os.W_OK):
        return Check(name, False, f"not writable: {path}")
    return Check(name, True, str(path))


def _has_unsafe_path_controls(value: str) -> bool:
    return any(unicodedata.category(ch) in {"Cc", "Cf", "Cs"} for ch in value)


def _configured_path_text(value: object, *, field: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be text or Path")
    text = os.fspath(value)
    if not text or text != text.strip() or _has_unsafe_path_controls(text):
        raise ValueError(f"{field} must be non-empty unpadded text without controls")
    return text


def _persistence_parent_check(name: str, value: object) -> Check:
    try:
        text = _configured_path_text(value, field="configured path")
    except ValueError as exc:
        return Check(name, False, str(exc))
    try:
        path = Path(text).expanduser()
        if not path.name:
            return Check(name, False, "configured path must name a persistence file")
        resolved = path.resolve()
        if resolved.is_dir():
            return Check(name, False, "configured path must name a file, not a directory")
        parent = resolved.parent
    except (OSError, RuntimeError, ValueError) as exc:
        return Check(name, False, f"invalid path: {text} ({exc})")
    return _directory_check(name, parent)


def _http_url_check(name: str, value: object) -> Check:
    try:
        normalized = require_http_url(value)
        parsed = urlparse(normalized)
        if parsed.query or parsed.fragment:
            raise ValueError("service URL must not contain a query string or fragment")
    except ValueError as exc:
        return Check(name, False, str(exc))
    return Check(name, True, normalized)


def _feature_flag_check(name: str, env_name: str) -> Check:
    try:
        enabled = env_bool(env_name, False)
    except ValueError as exc:
        return Check(name, False, str(exc))
    return Check(name, True, "enabled" if enabled else "disabled")


def _single_line(value: object, *, limit: int = _DETAIL_LIMIT) -> str:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("diagnostic detail limit must be a positive integer")
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001
        rendered = value.__class__.__name__
    return single_line_text(rendered, limit=limit)


def _check_name(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("check names must be non-empty unpadded text")
    if len(value) > 80:
        raise ValueError("check names must be at most 80 characters")
    if any(ch in _BIDI_CONTROLS for ch in value):
        raise ValueError("check names must not contain bidirectional controls")
    normalized = _single_line(value, limit=80)
    if not normalized:
        raise ValueError("check names must contain visible text")
    return normalized


def _validated_checks(checks: object) -> list[tuple[Check, str]]:
    if not isinstance(checks, (list, tuple)):
        raise ValueError("checks must be a list or tuple")
    if len(checks) > _MAX_CHECKS:
        raise ValueError(f"checks must contain at most {_MAX_CHECKS} values")
    seen: dict[str, str] = {}
    validated: list[tuple[Check, str]] = []
    for check in checks:
        if not isinstance(check, Check):
            raise ValueError("checks must contain Check values")
        name = _check_name(check.name)
        if not isinstance(check.ok, bool):
            raise ValueError("check status must be boolean")
        identity = unicodedata.normalize("NFKC", name).casefold()
        if identity in seen:
            raise ValueError(f"duplicate diagnostic check name: {name} conflicts with {seen[identity]}")
        seen[identity] = name
        validated.append((check, name))
    return validated


def _diagnostic_root(base_dir: object) -> Path:
    if base_dir is None:
        return Path(__file__).resolve().parent
    try:
        text = _configured_path_text(base_dir, field="base_dir")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    try:
        root = Path(text).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"base_dir could not be resolved: {exc}") from exc
    if not root.is_dir():
        raise ValueError("base_dir must resolve to an existing directory")
    return root


def _runtime_config() -> tuple[object | None, Check]:
    try:
        module = importlib.import_module("config")
    except ValueError as exc:
        return None, Check("configuration", False, _single_line(exc) or "invalid configuration")
    except Exception as exc:  # noqa: BLE001
        return None, Check("configuration", False, f"configuration import failed ({exc.__class__.__name__})")
    missing = [field for field in _REQUIRED_CONFIG_FIELDS if not hasattr(module, field)]
    if missing:
        return None, Check("configuration", False, "missing fields: " + ", ".join(missing))
    return module, Check("configuration", True, "validated")


def collect_checks(base_dir: Path | str | None = None) -> list[Check]:
    root = _diagnostic_root(base_dir)
    checks = [
        _python_check(),
        Check("platform", platform.system() == "Darwin", platform.platform()),
        _model_check("pose-model", root / "pose_landmarker_lite.task"),
        _model_check("gesture-model", root / "gesture_recognizer.task"),
    ]
    runtime_config, config_check = _runtime_config()
    checks.append(config_check)
    if runtime_config is not None:
        checks.append(_persistence_parent_check("checkpoint-dir", runtime_config.CHECKPOINT_DB_PATH))
        checks.append(_persistence_parent_check("report-dir", runtime_config.DAILY_REPORT_PATH))
        checks.append(_http_url_check("ollama-url", runtime_config.OLLAMA_HOST))
        checks.append(_http_url_check("deepseek-url", runtime_config.DEEPSEEK_BASE_URL))
        key_configured = bool(runtime_config.DEEPSEEK_API_KEY)
        checks.append(Check("deepseek-key", True, "configured" if key_configured else "not configured (optional)"))
    checks.append(_feature_flag_check("tts", "WAKEUP_ALLOW_TTS"))
    checks.append(_feature_flag_check("browser-control", "WAKEUP_ALLOW_BROWSER_CONTROL"))
    checks.append(_feature_flag_check("external-messaging", "WAKEUP_ALLOW_EXTERNAL_MESSAGING"))
    checks.append(_feature_flag_check("process-control", "WAKEUP_ALLOW_PROCESS_CONTROL"))
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for check, name in _validated_checks(checks):
        marker = "OK" if check.ok else "WARN"
        lines.append(f"[{marker}] {name}: {_single_line(check.detail)}")
    return "\n".join(lines)


def checks_payload(checks: list[Check]) -> list[dict[str, object]]:
    return [
        {"name": name, "ok": check.ok, "detail": _single_line(check.detail)}
        for check, name in _validated_checks(checks)
    ]


def format_checks_json(checks: list[Check]) -> str:
    return json.dumps(checks_payload(checks), ensure_ascii=False, sort_keys=True, allow_nan=False)


def diagnostics_exit_code(checks: list[Check]) -> int:
    return 1 if any(
        not check.ok
        and unicodedata.normalize("NFKC", name).casefold() in _CRITICAL_IDENTITIES
        for check, name in _validated_checks(checks)
    ) else 0
