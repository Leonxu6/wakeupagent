"""Offline diagnostics for common WakeUpAgent setup problems."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import config
from runtime_paths import PROJECT_ROOT


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


_MODEL_FILES = ("pose_landmarker_lite.task", "gesture_recognizer.task")


def collect_diagnostics(root: Path = PROJECT_ROOT) -> list[CheckResult]:
    """Return deterministic, network-free setup checks."""
    results: list[CheckResult] = []

    py_ok = sys.version_info >= (3, 12)
    results.append(
        CheckResult(
            "python",
            py_ok,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    results.append(
        CheckResult(
            "platform",
            platform.system() == "Darwin",
            platform.system() or "unknown",
            required=False,
        )
    )

    for filename in _MODEL_FILES:
        path = root / filename
        results.append(CheckResult(f"model:{filename}", path.is_file(), str(path)))

    results.append(
        CheckResult(
            "deepseek_api_key",
            bool(config.DEEPSEEK_API_KEY.strip()),
            "configured" if config.DEEPSEEK_API_KEY.strip() else "missing",
            required=False,
        )
    )

    enabled = [
        name
        for name, value in (
            ("wechat", config.ENABLE_WECHAT_ACTIONS),
            ("app_termination", config.ENABLE_APP_TERMINATION),
            ("chaos", config.ENABLE_CHAOS_ACTIONS),
        )
        if value
    ]
    results.append(
        CheckResult(
            "disruptive_actions",
            not enabled,
            "disabled" if not enabled else "enabled: " + ", ".join(enabled),
            required=False,
        )
    )

    return results


def required_checks_pass(results: list[CheckResult]) -> bool:
    """Return whether every required diagnostic passed."""
    return all(result.ok for result in results if result.required)
