"""Fast, side-effect-free diagnostics for WakeUpAgent installations."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import config


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _model_check(name: str, path: Path) -> Check:
    if not path.exists():
        return Check(name, False, f"missing: {path}")
    if not path.is_file():
        return Check(name, False, f"not a file: {path}")
    if path.stat().st_size == 0:
        return Check(name, False, f"empty model file: {path}")
    return Check(name, True, f"{path.name} ({path.stat().st_size} bytes)")


def collect_checks(base_dir: Path | None = None) -> list[Check]:
    """Collect checks without opening the camera or contacting network services."""
    root = base_dir or Path(__file__).resolve().parent
    checks = [
        Check("python", True, platform.python_version()),
        Check("platform", platform.system() == "Darwin", platform.platform()),
        _model_check("pose-model", root / "pose_landmarker_lite.task"),
        _model_check("gesture-model", root / "gesture_recognizer.task"),
    ]

    checkpoint_parent = Path(config.CHECKPOINT_DB_PATH).expanduser().resolve().parent
    report_parent = Path(config.DAILY_REPORT_PATH).expanduser().resolve().parent
    checks.append(Check("checkpoint-dir", checkpoint_parent.exists(), str(checkpoint_parent)))
    checks.append(Check("report-dir", report_parent.exists(), str(report_parent)))

    ollama = urlparse(config.OLLAMA_HOST)
    checks.append(Check("ollama-url", bool(ollama.hostname), config.OLLAMA_HOST))
    deepseek = urlparse(config.DEEPSEEK_BASE_URL)
    checks.append(Check("deepseek-url", bool(deepseek.hostname), config.DEEPSEEK_BASE_URL))
    checks.append(
        Check(
            "deepseek-key",
            bool(config.DEEPSEEK_API_KEY),
            "configured" if config.DEEPSEEK_API_KEY else "not configured",
        )
    )
    return checks


def format_checks(checks: list[Check]) -> str:
    """Render a deterministic human-readable diagnostics report."""
    lines = []
    for check in checks:
        marker = "OK" if check.ok else "WARN"
        lines.append(f"[{marker}] {check.name}: {check.detail}")
    return "\n".join(lines)


def diagnostics_exit_code(checks: list[Check]) -> int:
    """Return non-zero only for installation-critical model failures."""
    critical = {"pose-model", "gesture-model"}
    return 1 if any(not c.ok and c.name in critical for c in checks) else 0
