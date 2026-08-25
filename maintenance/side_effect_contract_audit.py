"""Ensure side-effecting tools retain explicit opt-in gates and safe registration."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from maintenance.common import print_failures, require_root

_REQUIRED_GATES = {
    "play_tts_punishment": "WAKEUP_ALLOW_TTS",
    "send_wechat_shame_message": "WAKEUP_ALLOW_EXTERNAL_MESSAGING",
    "open_webpage": "WAKEUP_ALLOW_BROWSER_CONTROL",
    "force_close_app": "WAKEUP_ALLOW_PROCESS_CONTROL",
}


def audit(root: Path) -> list[str]:
    root = require_root(root)
    path = root / "tools.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [f"tools.py: could not parse side-effect tools ({exc})"]

    functions = {node.name: ast.get_source_segment(path.read_text(encoding="utf-8"), node) or "" for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    failures: list[str] = []
    for function_name, env_name in _REQUIRED_GATES.items():
        source = functions.get(function_name)
        if source is None:
            failures.append(f"tools.py: missing side-effect tool {function_name}")
        elif f'_feature_enabled("{env_name}")' not in source:
            failures.append(f"tools.py: {function_name} must require {env_name}")

    module_source = path.read_text(encoding="utf-8")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ALL_TOOLS" for target in node.targets):
            segment = ast.get_source_segment(module_source, node) or ""
            if "chaos_terminal_punishment" in segment:
                failures.append("tools.py: legacy chaos tool must not be registered in ALL_TOOLS")
            break
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
