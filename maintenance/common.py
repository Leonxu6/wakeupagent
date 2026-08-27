"""Shared primitives for repository maintenance audits."""
from __future__ import annotations

import subprocess
from pathlib import Path

TEXT_EXTENSIONS = {
    ".md", ".py", ".toml", ".yml", ".yaml", ".txt", ".json",
    ".ini", ".cfg", ".sh", ".example",
}
IGNORED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "build", "dist",
}
NON_RUNTIME_ROOTS = {"tests", "maintenance"}


def require_root(root: object) -> Path:
    """Return a validated repository root."""
    if not isinstance(root, Path):
        raise ValueError("root must be a pathlib.Path")
    if not root.exists():
        raise ValueError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    return root


def relative_files(root: Path, *, suffixes: set[str] | None = None) -> list[Path]:
    """Return deterministic repository-relative files, excluding generated trees."""
    root = require_root(root)
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in rel.parts):
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        files.append(rel)
    return sorted(files)


def tracked_files(root: Path) -> list[Path]:
    """Return tracked repository paths using Git's authoritative index."""
    root = require_root(root)
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ValueError("could not enumerate tracked repository files") from exc
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("tracked file list is not valid UTF-8") from exc
    paths = [Path(item) for item in decoded.split("\0") if item]
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise ValueError("tracked file list contained a path outside repository")
    return paths


def production_python_files(root: Path) -> list[Path]:
    """Return tracked runtime Python files, excluding tests and maintenance tooling."""
    root = require_root(root)
    return [
        rel
        for rel in tracked_files(root)
        if rel.suffix == ".py" and rel.parts and rel.parts[0] not in NON_RUNTIME_ROOTS
    ]


def print_failures(failures: list[str]) -> int:
    """Print audit failures and return a conventional process exit code."""
    for failure in failures:
        print(failure)
    return 1 if failures else 0
