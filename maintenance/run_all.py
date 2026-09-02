"""Run the repository's maintenance audits as one deterministic local/CI command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_AUDITS = (
    "ci_contract_audit.py",
    "env_example_audit.py",
    "python_version_audit.py",
    "model_asset_audit.py",
    "python_compile_audit.py",
    "text_integrity_audit.py",
    "repository_hygiene_audit.py",
    "docs_contract_audit.py",
    "env_parity_audit.py",
    "gitignore_contract_audit.py",
    "workflow_security_audit.py",
    "side_effect_contract_audit.py",
    "cli_contract_audit.py",
    "pyproject_metadata_audit.py",
    "dependency_lock_audit.py",
    "readme_command_audit.py",
    "path_case_audit.py",
    "secret_filename_audit.py",
    "test_layout_audit.py",
    "source_import_audit.py",
    "json_syntax_audit.py",
    "toml_syntax_audit.py",
    "unicode_bidi_audit.py",
    "dynamic_code_audit.py",
    "subprocess_shell_audit.py",
    "unsafe_deserialization_audit.py",
    "weak_hash_audit.py",
    "tls_verification_audit.py",
    "tempfile_safety_audit.py",
    "dependency_source_audit.py",
    "markdown_fence_audit.py",
    "bare_except_audit.py",
    "mutable_default_audit.py",
    "wildcard_import_audit.py",
    "debug_call_audit.py",
    "os_system_audit.py",
    "datetime_utcnow_audit.py",
    "absolute_user_path_audit.py",
    "duplicate_definition_audit.py",
    "unsafe_chmod_audit.py",
    "http_timeout_audit.py",
    "urlopen_timeout_audit.py",
    "subprocess_timeout_audit.py",
    "async_blocking_sleep_audit.py",
    "async_subprocess_audit.py",
    "runtime_assert_audit.py",
    "baseexception_handler_audit.py",
    "sys_path_mutation_audit.py",
    "interactive_input_audit.py",
    "unverified_ssl_context_audit.py",
    "uuid1_audit.py",
    "socket_timeout_audit.py",
    "path_text_encoding_audit.py",
    "open_text_encoding_audit.py",
    "naive_datetime_now_audit.py",
    "naive_fromtimestamp_audit.py",
    "builtin_hash_audit.py",
    "archive_extract_audit.py",
    "unpack_archive_audit.py",
    "os_chdir_audit.py",
    "os_umask_audit.py",
    "locale_mutation_audit.py",
    "warning_suppression_audit.py",
    "socket_default_timeout_audit.py",
    "signal_handler_audit.py",
    "sys_exit_audit.py",
    "recursion_limit_audit.py",
    "gc_disable_audit.py",
    "random_seed_audit.py",
    "asyncio_run_audit.py",
    "environ_mutation_audit.py",
    "logging_basic_config_audit.py",
    "json_nan_audit.py",
    "sql_interpolation_audit.py",
    "subprocess_run_check_audit.py",
    "thread_daemon_audit.py",
)

# Repository-wide rules start as visible advisories so the project can inspect
# legacy findings before promoting a clean rule to a blocking CI gate.
_ADVISORY_AUDITS = (
    "contextlib_suppress_audit.py",
    "tempfile_mktemp_audit.py",
    "unbounded_queue_audit.py",
    "unbounded_deque_audit.py",
    "unbounded_lru_cache_audit.py",
    "os_putenv_audit.py",
    "tzset_audit.py",
    "numpy_global_state_audit.py",
    "resource_limits_audit.py",
    "signal_timers_audit.py",
    "sqlite_timeout_audit.py",
    "httpx_timeout_audit.py",
    "aiohttp_timeout_audit.py",
    "websocket_timeout_audit.py",
    "requests_session_audit.py",
    "asyncio_create_task_audit.py",
    "multiprocessing_daemon_audit.py",
    "shelve_usage_audit.py",
    "zip_extractall_audit.py",
    "atexit_register_audit.py",
    "multiprocessing_start_method_audit.py",
    "multiprocessing_executable_audit.py",
    "multiprocessing_forkserver_preload_audit.py",
    "asyncio_event_loop_policy_audit.py",
    "asyncio_set_event_loop_audit.py",
    "sys_excepthook_audit.py",
    "sys_trace_audit.py",
    "sys_profile_audit.py",
    "sys_int_max_str_digits_audit.py",
    "sys_coroutine_origin_tracking_audit.py",
    "sys_asyncgen_hooks_audit.py",
    "threading_excepthook_audit.py",
    "threading_stack_size_audit.py",
    "threading_trace_all_audit.py",
    "threading_profile_all_audit.py",
    "gc_debug_audit.py",
    "gc_threshold_audit.py",
    "random_setstate_audit.py",
    "tempfile_tempdir_audit.py",
    "urllib_install_opener_audit.py",
    "sqlite_callback_tracebacks_audit.py",
    "sqlite_shared_cache_audit.py",
    "logging_capture_warnings_audit.py",
    "numpy_printoptions_audit.py",
    "decimal_default_context_audit.py",
    "runtime_global_state_audit.py",
)


def _validate_scripts(scripts: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(scripts, tuple):
        raise ValueError("scripts must be a tuple of audit filenames")
    seen: set[str] = set()
    for script in scripts:
        if (
            not isinstance(script, str)
            or not script
            or script != script.strip()
            or not script.endswith(".py")
            or "/" in script
            or "\\" in script
        ):
            raise ValueError("audit script names must be simple .py filenames")
        if script in seen:
            raise ValueError("audit script names must be unique")
        seen.add(script)
    return scripts


def _failure_detail(result: object) -> str:
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    detail = " ".join(f"{stdout}\n{stderr}".split()) or "audit failed"
    return detail[:1000]


def _run_audit(root: Path, script: str) -> str | None:
    path = root / "maintenance" / script
    if not path.is_file():
        return "audit script is missing"
    module = f"maintenance.{Path(script).stem}"
    try:
        result = subprocess.run(
            [sys.executable, "-m", module, str(root)],
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "audit timed out"
    except OSError as exc:
        return f"audit could not start ({exc})"
    return _failure_detail(result) if result.returncode else None


def _run_selected(root: Path, scripts: tuple[str, ...]) -> list[str]:
    if not isinstance(root, Path) or not root.is_dir():
        raise ValueError("root must be an existing directory")
    scripts = _validate_scripts(scripts)
    findings: list[str] = []
    for script in scripts:
        detail = _run_audit(root, script)
        if detail:
            findings.append(f"{script}: {detail}")
    return findings


def run_audits(root: Path, *, scripts: tuple[str, ...] = _AUDITS) -> list[str]:
    return _run_selected(root, scripts)


def run_advisory_audits(
    root: Path, *, scripts: tuple[str, ...] = _ADVISORY_AUDITS
) -> list[str]:
    return _run_selected(root, scripts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root)
    failures = run_audits(root)
    advisories = run_advisory_audits(root)
    for finding in advisories:
        print(f"[advisory] {finding}", file=sys.stderr)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
