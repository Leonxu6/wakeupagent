"""
main.py — Cyber-Superego entry point

usage:
    uv run main.py           # perception loop (live camera)
    uv run main.py --graph   # one-shot langgraph run (mock)
"""
import argparse
from datetime import datetime, date

from rich.console import Console
from config import LOG_A, LOG_B, LOG_C

console = Console()

# 固定 thread_id：让 checkpointer 跨轮累积同一用户的状态
_THREAD_CONFIG = {"configurable": {"thread_id": "superego_main"}}


def run_perception_mode():
    from perception import run_perception_loop
    from graph import build_graph

    graph = build_graph()
    last_summary = [""]  # 大脑 context 同步给小脑用

    def on_vision(text: str, ts: str, is_healthy: bool, should_escalate: bool):
        state = {
            "current_vision_text": text,
            "healthy": is_healthy,
            "should_escalate": should_escalate,
            "timestamp": ts,
            "session_date": date.today().isoformat(),
        }
        for update in graph.stream(state, config=_THREAD_CONFIG, stream_mode="updates"):
            for node_output in update.values():
                if isinstance(node_output, dict) and node_output.get("conversation_summary"):
                    last_summary[0] = node_output["conversation_summary"]

    run_perception_loop(
        state_callback=on_vision,
        get_summary=lambda: last_summary[0],
    )


def run_graph_mode():
    from graph import build_graph
    console.print(f"{LOG_A} building langgraph")
    graph = build_graph()
    state = {
        "current_vision_text": "person lying in bed scrolling phone",
        "healthy": False,
        "should_escalate": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_date": date.today().isoformat(),
    }
    console.print(f"{LOG_A} START -> [R] -> [A] -> [B] -> [C]")
    for _ in graph.stream(state, config=_THREAD_CONFIG, stream_mode="updates"):
        pass
    console.print(f"{LOG_A} graph run complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true", help="run mock graph flow")
    args = ap.parse_args()

    console.print("[cyan]CYBER-SUPEREGO[/cyan]  edge-cloud hybrid supervisor")
    console.print(f"  nodes: [R] reset  [A] perception+cerebellum  [B] decision  [C] execution")
    console.print(f"  stack: mediapipe / moondream / qwen2.5(cerebellum) / deepseek / langgraph\n")

    if args.graph:
        run_graph_mode()
    else:
        run_perception_mode()


if __name__ == "__main__":
    main()
