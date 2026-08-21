"""
main.py — Cyber-Superego entry point

usage:
    uv run main.py           # perception loop (live camera)
    uv run main.py --graph   # one-shot langgraph run (mock)
    uv run main.py --doctor  # network-free setup diagnostics
"""
import argparse
from datetime import datetime

from rich.console import Console
from config import LOG_A, LOG_B, LOG_C

console = Console()

_THREAD_CONFIG = {"configurable": {"thread_id": "superego_main"}}
_CONTEXT_WINDOW = 15


def _observation_state(text: str, ts: str, is_healthy: bool, should_escalate: bool) -> dict:
    """Build one graph input without overwriting checkpointed daily state."""
    return {
        "current_vision_text": text,
        "healthy": is_healthy,
        "should_escalate": should_escalate,
        "timestamp": ts,
    }


def _build_context(summary: str, recent_items: list[str], window: int = _CONTEXT_WINDOW) -> str:
    """Build the bounded text context shared with the local classifier."""
    parts = []
    if summary:
        parts.append(f"Summary: {summary[:200]}")
    if recent_items:
        parts.append("Recent history:\n" + "\n".join(recent_items[-window:]))
    return "\n\n".join(parts)


def run_perception_mode():
    from perception import run_perception_loop
    from graph import build_graph

    graph = build_graph()
    last_summary = [""]
    recent_items: list[str] = []

    def _stream_graph(state: dict):
        """运行图并更新 recent_items / last_summary。"""
        try:
            for update in graph.stream(state, config=_THREAD_CONFIG, stream_mode="updates"):
                for node_output in update.values():
                    if not isinstance(node_output, dict):
                        continue
                    if node_output.get("conversation_summary"):
                        last_summary[0] = node_output["conversation_summary"]
                    for m in node_output.get("messages", []):
                        if getattr(m, "type", None) == "ai" and getattr(m, "content", ""):
                            recent_items.append(f"[Brain] {m.content[:120]}")
        except RuntimeError:
            pass
        except Exception as e:
            console.print(f"[red]stream error: {e}[/red]")

        while len(recent_items) > _CONTEXT_WINDOW:
            recent_items.pop(0)

    def on_vision(text: str, ts: str, is_healthy: bool, should_escalate: bool):
        recent_items.append(f"[Obs] {text[:100]}")
        state = _observation_state(text, ts, is_healthy, should_escalate)
        _stream_graph(state)

    def get_context() -> str:
        return _build_context(last_summary[0], recent_items)

    run_perception_loop(
        state_callback=on_vision,
        get_context=get_context,
    )


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


def run_doctor_mode() -> int:
    """Print offline setup diagnostics and return a process-style status code."""
    from doctor import collect_diagnostics, required_checks_pass

    results = collect_diagnostics()
    for result in results:
        if result.ok:
            marker = "[green]PASS[/green]"
        elif result.required:
            marker = "[red]FAIL[/red]"
        else:
            marker = "[yellow]WARN[/yellow]"
        console.print(f"{marker} {result.name}: {result.detail}")

    if required_checks_pass(results):
        console.print("[green]required checks passed[/green]")
        return 0
    console.print("[red]one or more required checks failed[/red]")
    return 1


def main():
    ap = argparse.ArgumentParser(description="WakeUpAgent edge-cloud productivity supervisor")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--graph", action="store_true", help="run mock graph flow")
    mode.add_argument("--doctor", action="store_true", help="run offline setup diagnostics")
    args = ap.parse_args()

    console.print("[cyan]CYBER-SUPEREGO[/cyan]  edge-cloud hybrid supervisor")
    console.print(f"  nodes: [R] reset  [A] perception+cerebellum  [B] decision  [C] execution")
    console.print(f"  stack: mediapipe / moondream / qwen2.5(cerebellum) / deepseek / langgraph\n")

    if args.doctor:
        raise SystemExit(run_doctor_mode())
    if args.graph:
        run_graph_mode()
    else:
        run_perception_mode()


if __name__ == "__main__":
    main()
