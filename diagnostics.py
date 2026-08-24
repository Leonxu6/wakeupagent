"""Fast, side-effect-free diagnostics for WakeUpAgent installations."""
from __future__ import annotations

import importlib
import json
import os
import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from safety import require_http_url
from settings import env_bool


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


def _persistence_parent_check(name: str, value: object) -> Check:
    if not isinstance(value, (str, Path)):
        return Check(name, False, "configured path must be text or Path")
    try:
        parent = Path(value).expanduser().resolve().parent
    except (OSError, RuntimeError, ValueError) as exc:
        return Check(name, False, f"invalid path: {value} ({exc})")
    return _directory_check(name, parent)


def _http_url_check(name: str, value: object) -> Check:
    try:
        normalized = require_http_url(value)
    except ValueError as exc:
        return Check(name, False, str(exc))
    return Check(name, True, normalized)


def _feature_flag_check(name: str, env_name: str) -> Check:
    try:
        enabled = env_bool(env_name, False)
    except ValueError as exc:
        return Check(name, False, str(exc))
    return Check(name, True, "enabled" if enabled else "disabled")


def _single_line(value: object) -> str:
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001
        rendered = value.__class__.__name__
    return " ".join(rendered.split())


def _diagnostic_root(base_dir: object) -> Path:
    if base_dir is None:
        return Path(__file__).resolve().parent
    if not isinstance(base_dir, (str, Path)):
        raise ValueError("base_dir must be a path string or Path")
    return Path(base_dir).expanduser()


def _runtime_config() -> tuple[object | None, Check]:
    try:
        module = importlib.import_module("config")
    except Exception as exc:  # noqa: BLE001
        return None, Check("configuration", False, _single_line(exc) or exc.__class__.__name__)
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
        checks.append(
            Check(
                "deepseek-key",
                bool(runtime_config.DEEPSEEK_API_KEY),
                "configured" if runtime_config.DEEPSEEK_API_KEY else "not configured",
            )
        )
    checks.append(_feature_flag_check("tts", "WAKEUP_ALLOW_TTS"))
    checks.append(_feature_flag_check("browser-control", "WAKEUP_ALLOW_BROWSER_CONTROL"))
    checks.append(_feature_flag_check("external-messaging", "WAKEUP_ALLOW_EXTERNAL_MESSAGING"))
    checks.append(_feature_flag_check("process-control", "WAKEUP_ALLOW_PROCESS_CONTROL"))
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        marker = "OK" if check.ok else "WARN"
        lines.append(f"[{marker}] {_single_line(check.name)}: {_single_line(check.detail)}")
    return "\n".join(lines)


def checks_payload(checks: list[Check]) -> list[dict[str, object]]:
    return [
        {"name": _single_line(check.name), "ok": bool(check.ok), "detail": _single_line(check.detail)}
        for check in checks
    ]


def format_checks_json(checks: list[Check]) -> str:
    return json.dumps(checks_payload(checks), ensure_ascii=False, sort_keys=True)


def diagnostics_exit_code(checks: list[Check]) -> int:
    return 1 if any(not c.ok and c.name in _CRITICAL_CHECKS for c in checks) else 0
