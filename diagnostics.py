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


def _persistence_parent_check(name: str, value: object) -> Check:
    """Resolve a configured persistence path without letting path errors abort diagnostics."""
    if not isinstance(value, (str, Path)):
        return Check(name, False, "configured path must be text or Path")
    try:
        parent = Path(value).expanduser().resolve().parent
    except (OSError, RuntimeError, ValueError) as exc:
        return Check(name, False, f"invalid path: {value} ({exc})")
    return _directory_check(name, parent)


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
    """Keep diagnostics machine-readable even when rendering or OS errors misbehave."""
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001
        rendered = value.__class__.__name__
    return " ".join(rendered.split())


def _diagnostic_root(base_dir: object) -> Path:
    """Normalize an optional diagnostics root without accepting accidental scalar values."""
    if base_dir is None:
        return Path(__file__).resolve().parent
    if not isinstance(base_dir, (str, Path)):
        raise ValueError("base_dir must be a path string or Path")
    return Path(base_dir).expanduser()


def collect_checks(base_dir: Path | str | None = None) -> list[Check]:
    """Collect checks without opening the camera or contacting network services."""
    root = _diagnostic_root(base_dir)
    checks = [
        _python_check(),
        Check("platform", platform.system() == "Darwin", platform.platform()),
        _model_check("pose-model", root / "pose_landmarker_lite.task"),
        _model_check("gesture-model", root / "gesture_recognizer.task"),
    ]

    checks.append(_persistence_parent_check("checkpoint-dir", config.CHECKPOINT_DB_PATH))
    checks.append(_persistence_parent_check("report-dir", config.DAILY_REPORT_PATH))

    checks.append(_http_url_check("ollama-url", config.OLLAMA_HOST))
    checks.append(_http_url_check("deepseek-url", config.DEEPSEEK_BASE_URL))
    checks.append(
        Check(
            "deepseek-key",
            bool(config.DEEPSEEK_API_KEY),
            "configured" if config.DEEPSEEK_API_KEY else "not configured",
        )
    )
    checks.append(_feature_flag_check("tts", "WAKEUP_ALLOW_TTS"))
    checks.append(_feature_flag_check("browser-control", "WAKEUP_ALLOW_BROWSER_CONTROL"))
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
    """Return non-zero when the interpreter or required local models are unusable."""
    critical = {"python", "pose-model", "gesture-model"}
    return 1 if any(not c.ok and c.name in critical for c in checks) else 0
