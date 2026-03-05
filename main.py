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

# 同步给小脑的滚动上下文：保留最近 N 条记录（观察文本 + 大脑判决）
_CONTEXT_WINDOW = 15


def run_perception_mode():
    from perception import run_perception_loop
    from graph import build_graph

    graph = build_graph()
    last_summary = [""]          # 压缩摘要（长期记忆）
    recent_items: list[str] = [] # 近期观察+判决 滚动列表（短期记忆）

    def on_vision(text: str, ts: str, is_healthy: bool, should_escalate: bool):
        # 把本次感知观察追加进上下文（大脑处理前先记录）
        recent_items.append(f"[Obs] {text[:100]}")

        state = {
            "current_vision_text": text,
            "healthy": is_healthy,
            "should_escalate": should_escalate,
            "timestamp": ts,
            "session_date": date.today().isoformat(),
        }
        for update in graph.stream(state, config=_THREAD_CONFIG, stream_mode="updates"):
            for node_output in update.values():
                if not isinstance(node_output, dict):
                    continue
                # 更新压缩摘要
                if node_output.get("conversation_summary"):
                    last_summary[0] = node_output["conversation_summary"]
                # 捕获大脑的 AI 消息（判决文本），追加到滚动窗口
                for m in node_output.get("messages", []):
                    if getattr(m, "type", None) == "ai" and getattr(m, "content", ""):
                        recent_items.append(f"[Brain] {m.content[:120]}")

        # 保持滚动窗口大小
        while len(recent_items) > _CONTEXT_WINDOW:
            recent_items.pop(0)

    def get_context() -> str:
        """返回给小脑的完整上下文字符串：摘要 + 近期记录。"""
        parts = []
        if last_summary[0]:
            parts.append(f"Summary: {last_summary[0][:200]}")
        if recent_items:
            parts.append("Recent history:\n" + "\n".join(recent_items[-10:]))
        return "\n\n".join(parts)

    run_perception_loop(
        state_callback=on_vision,
        get_context=get_context,
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
