"""Detect process-wide runtime mutations that can surprise long-running agents.

The checks in this module are intentionally advisory. They focus on APIs that
change interpreter- or process-global state rather than local object state, so
a call in one component can silently alter unrelated components later.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from maintenance.ast_rules import call_name, iter_calls
from maintenance.common import print_failures, production_python_files, require_root

_RULE_MESSAGES = {
    "faulthandler.enable": "faulthandler configuration is process-wide",
    "faulthandler.disable": "faulthandler configuration is process-wide",
    "faulthandler.register": "faulthandler signal registration is process-wide",
    "faulthandler.unregister": "faulthandler signal registration is process-wide",
    "tracemalloc.start": "tracemalloc lifecycle is process-wide",
    "tracemalloc.stop": "tracemalloc lifecycle is process-wide",
    "tracemalloc.clear_traces": "tracemalloc traces are process-wide",
    "os.nice": "process priority changes affect the entire agent process",
    "os.setpriority": "process priority changes affect scheduler behavior",
    "os.register_at_fork": "fork hooks persist for the process lifetime",
    "signal.set_wakeup_fd": "signal wakeup routing is process-wide",
    "signal.siginterrupt": "signal syscall restart behavior is process-wide",
    "signal.pthread_sigmask": "signal masks affect thread/process delivery semantics",
    "threading.settrace": "default tracing affects subsequently created threads",
    "threading.setprofile": "default profiling affects subsequently created threads",
    "logging.disable": "logging.disable changes process-wide logging visibility",
    "warnings.resetwarnings": "resetwarnings replaces process-wide warning filters",
    "cv2.setNumThreads": "OpenCV thread-pool size is global process state",
    "cv2.setRNGSeed": "OpenCV RNG seed changes shared native-library state",
}


def findings_for_source(source: str, *, path: str = "<memory>") -> list[str]:
    """Return deterministic findings for one Python source string."""
    if not isinstance(source, str):
        raise ValueError("source must be text")
    if not isinstance(path, str) or not path or path != path.strip():
        raise ValueError("path must be clean non-empty text")
    findings: list[str] = []
    for call in iter_calls(source):
        name = call_name(call)
        detail = _RULE_MESSAGES.get(name or "")
        if detail:
            findings.append(f"{path}:{call.lineno}: {name}: {detail}")
    return findings


def audit(root: Path) -> list[str]:
    root = require_root(root)
    findings: list[str] = []
    for rel in production_python_files(root):
        path = root / rel
        if path.is_symlink():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(f"{rel}: unreadable source ({exc.__class__.__name__})")
            continue
        findings.extend(findings_for_source(source, path=rel.as_posix()))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
