"""
main.py — Cyber-Superego entry point

usage:
    uv run main.py           # perception loop (live camera)
    uv run main.py --graph   # one-shot langgraph run (mock)
"""
import argparse
from datetime import datetime

from rich.console import Console
from config import LOG_A, LOG_B, LOG_C

console = Console()


def run_perception_mode():
    from perception import run_perception_loop

    def on_vision(text: str, ts: str):
        console.print(f"{LOG_A} state_callback -> graph trigger would go here")

    run_perception_loop(state_callback=on_vision)


def run_graph_mode():
    from graph import build_graph
    console.print(f"{LOG_A} building langgraph")
    graph = build_graph()
    state = {
        "messages": [],
        "current_vision_text": "person lying in bed scrolling phone",
        "healthy": False,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    console.print(f"{LOG_A} START -> [A] -> [B] -> [C]")
    for _ in graph.stream(state, stream_mode="values"):
        pass
    console.print(f"{LOG_A} graph run complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true", help="run mock graph flow")
    args = ap.parse_args()

    console.print("[cyan]CYBER-SUPEREGO[/cyan]  edge-cloud hybrid supervisor")
    console.print(f"  nodes: [A] perception  [B] decision  [C] execution")
    console.print(f"  stack: mediapipe / moondream / deepseek / langgraph\n")

    if args.graph:
        run_graph_mode()
    else:
        run_perception_mode()


if __name__ == "__main__":
    main()
