"""
main.py — Cyber-Superego entry point

usage:
    uv run main.py           # perception loop (live camera)
    uv run main.py --graph   # one-shot langgraph run (mock)
    uv run main.py --check   # validate local installation without camera/network calls
"""
import argparse
from datetime import datetime

from rich.console import Console

from config import LOG_A
from history import ContextHistory

console = Console()

# 固定 thread_id：让 checkpointer 跨轮累积同一用户的状态
_THREAD_CONFIG = {"configurable": {"thread_id": "superego_main"}}


def _observation_state(text: str, ts: str, is_healthy: bool, should_escalate: bool) -> dict:
    """Build one graph input without overwriting checkpointed daily state."""
    return {
        "current_vision_text": text,
        "healthy": is_healthy,
        "should_escalate": should_escalate,
        "timestamp": ts,
    }


def _is_shutdown_runtime_error(exc: RuntimeError) -> bool:
    """Recognize the executor shutdown race that can happen while quitting."""
    message = str(exc).lower()
    return "cannot schedule new futures after" in message and "shutdown" in message


def run_perception_mode():
    from perception import run_perception_loop
    from graph import build_graph

    graph = build_graph()
    history = ContextHistory(max_items=15)

    def _stream_graph(state: dict):
        """Run the graph and feed bounded summaries/decisions back to perception."""
        try:
            for update in graph.stream(state, config=_THREAD_CONFIG, stream_mode="updates"):
                for node_output in update.values():
                    if not isinstance(node_output, dict):
                        continue
                    if node_output.get("conversation_summary"):
                        history.set_summary(node_output["conversation_summary"])
                    for message in node_output.get("messages", []):
                        if getattr(message, "type", None) == "ai":
                            history.add_decision(getattr(message, "content", ""))
        except RuntimeError as exc:
            if not _is_shutdown_runtime_error(exc):
                console.print(f"[red]stream runtime error: {exc}[/red]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]stream error: {exc}[/red]")

    def on_vision(text: str, ts: str, is_healthy: bool, should_escalate: bool):
        """定时摄像头触发的感知回调。"""
        history.add_observation(text)
        _stream_graph(_observation_state(text, ts, is_healthy, should_escalate))

    def get_context() -> str:
        """Return the bounded summary and recent observations/decisions."""
        return history.render(recent=10)

    run_perception_loop(state_callback=on_vision, get_context=get_context)


def run_graph_mode():
    from graph import build_graph

    console.print(f"{LOG_A} building langgraph")
    graph = build_graph()
    state = _observation_state(
        "person lying in bed scrolling phone",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        False,
        True,
    )
    console.print(f"{LOG_A} START -> [R] -> [A] -> [B] -> [C]")
    for _ in graph.stream(state, config=_THREAD_CONFIG, stream_mode="updates"):
        pass
    console.print(f"{LOG_A} graph run complete")


def run_check_mode() -> int:
    """Print installation diagnostics without starting camera or network clients."""
    from diagnostics import collect_checks, diagnostics_exit_code, format_checks

    checks = collect_checks()
    console.print(format_checks(checks))
    return diagnostics_exit_code(checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WakeUpAgent edge-cloud productivity supervisor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--graph", action="store_true", help="run mock graph flow")
    mode.add_argument("--check", action="store_true", help="validate installation without camera/network side effects")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return run_check_mode()

    console.print("[cyan]CYBER-SUPEREGO[/cyan]  edge-cloud hybrid supervisor")
    console.print("  nodes: [R] reset  [A] perception+cerebellum  [B] decision  [C] execution")
    console.print("  stack: mediapipe / moondream / qwen2.5(cerebellum) / deepseek / langgraph\n")

    if args.graph:
        run_graph_mode()
    else:
        run_perception_mode()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
