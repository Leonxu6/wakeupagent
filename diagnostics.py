"""Fast, side-effect-free diagnostics for WakeUpAgent installations."""
from __future__ import annotations

import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import config
from safety import require_http_url
from settings import env_bool


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _python_check(version: tuple[int, int] | None = None) -> Check:
    """Verify the interpreter matches the project's declared Python floor."""
    current = version or (sys.version_info.major, sys.version_info.minor)
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
    """Report whether a configured persistence parent is an existing directory."""
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return Check(name, False, f"missing: {path}")
    except OSError as exc:
        return Check(name, False, f"unreadable: {path} ({exc})")
    if not stat.S_ISDIR(metadata.st_mode):
        return Check(name, False, f"not a directory: {path}")
    return Check(name, True, str(path))


def _http_url_check(name: str, value: object) -> Check:
    """Apply the same HTTP(S) boundary used by browser-facing runtime tools."""
    try:
        normalized = require_http_url(value)
    except ValueError as exc:
        return Check(name, False, str(exc))
    return Check(name, True, normalized)


def _feature_flag_check(name: str, env_name: str) -> Check:
    """Validate an opt-in side-effect flag without enabling the feature."""
    try:
        enabled = env_bool(env_name, False)
    except ValueError as exc:
        return Check(name, False, str(exc))
    return Check(name, True, "enabled" if enabled else "disabled")


def _single_line(value: object) -> str:
    """Keep diagnostics machine-readable even when an OS error contains newlines."""
    return " ".join(str(value).split())


def collect_checks(base_dir: Path | None = None) -> list[Check]:
    """Collect checks without opening the camera or contacting network services."""
    root = base_dir or Path(__file__).resolve().parent
    checks = [
        _python_check(),
        Check("platform", platform.system() == "Darwin", platform.platform()),
        _model_check("pose-model", root / "pose_landmarker_lite.task"),
        _model_check("gesture-model", root / "gesture_recognizer.task"),
    ]

    checkpoint_parent = Path(config.CHECKPOINT_DB_PATH).expanduser().resolve().parent
    report_parent = Path(config.DAILY_REPORT_PATH).expanduser().resolve().parent
    checks.append(_directory_check("checkpoint-dir", checkpoint_parent))
    checks.append(_directory_check("report-dir", report_parent))

    checks.append(_http_url_check("ollama-url", config.OLLAMA_HOST))
    checks.append(_http_url_check("deepseek-url", config.DEEPSEEK_BASE_URL))
    checks.append(
        Check(
            "deepseek-key",
            bool(config.DEEPSEEK_API_KEY),
            "configured" if config.DEEPSEEK_API_KEY else "not configured",
        )
    )
    checks.append(_feature_flag_check("external-messaging", "WAKEUP_ALLOW_EXTERNAL_MESSAGING"))
    checks.append(_feature_flag_check("process-control", "WAKEUP_ALLOW_PROCESS_CONTROL"))
    return checks


def format_checks(checks: list[Check]) -> str:
    """Render a deterministic one-line-per-check diagnostics report."""
    lines = []
    for check in checks:
        marker = "OK" if check.ok else "WARN"
        lines.append(f"[{marker}] {_single_line(check.name)}: {_single_line(check.detail)}")
    return "\n".join(lines)


def diagnostics_exit_code(checks: list[Check]) -> int:
    """Return non-zero only for installation-critical model failures."""
    critical = {"pose-model", "gesture-model"}
    return 1 if any(not c.ok and c.name in critical for c in checks) else 0
