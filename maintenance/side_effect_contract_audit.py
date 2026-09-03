"""Ensure side-effecting tools retain explicit opt-in gates and safe orchestration claims."""
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
_FORBIDDEN_ORCHESTRATION_CLAIMS = (
    "chaos_terminal_punishment",
    "50 个终端",
    "摧毁环境",
    "同时调用多个惩罚工具升级打击",
)


def _read(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeError) as exc:
        return None, f"{path.name}: could not read contract source ({exc.__class__.__name__})"


def audit(root: Path) -> list[str]:
    root = require_root(root)
    path = root / "tools.py"
    module_source, read_error = _read(path)
    if read_error:
        return [read_error]
    assert module_source is not None
    try:
        tree = ast.parse(module_source, filename=str(path))
    except SyntaxError:
        return ["tools.py: could not parse side-effect tools (SyntaxError)"]

    functions = {
        node.name: ast.get_source_segment(module_source, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    failures: list[str] = []
    for function_name, env_name in _REQUIRED_GATES.items():
        source = functions.get(function_name)
        if source is None:
            failures.append(f"tools.py: missing side-effect tool {function_name}")
        elif f'_feature_enabled("{env_name}")' not in source:
            failures.append(f"tools.py: {function_name} must require {env_name}")

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ALL_TOOLS" for target in node.targets):
            segment = ast.get_source_segment(module_source, node) or ""
            if "chaos_terminal_punishment" in segment:
                failures.append("tools.py: legacy chaos tool must not be registered in ALL_TOOLS")
            break

    graph_source, graph_error = _read(root / "graph.py")
    if graph_error:
        failures.append(graph_error)
    elif graph_source is not None:
        for claim in _FORBIDDEN_ORCHESTRATION_CLAIMS:
            if claim in graph_source:
                failures.append(f"graph.py: orchestration must not advertise removed/disruptive behavior: {claim}")

    side_effect_docs, docs_error = _read(root / "docs" / "side-effects.md")
    if docs_error:
        failures.append(docs_error)
    elif side_effect_docs is not None:
        required = ("legacy `chaos_terminal_punishment`", "intentionally inert", "not registered in `ALL_TOOLS`")
        for phrase in required:
            if phrase not in side_effect_docs:
                failures.append(f"docs/side-effects.md: missing legacy safety statement: {phrase}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    return print_failures(audit(Path(args.root)))


if __name__ == "__main__":
    raise SystemExit(main())
