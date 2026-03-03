"""
graph.py — LangGraph 状态机定义
包含：全局 State 定义、三大节点定义、图的构建逻辑
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from typing_extensions import TypedDict

from config import LOG_A, LOG_B, LOG_C
LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
from tools import ALL_TOOLS

console = Console()


# ── 全局状态定义 ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # LLM 对话历史 + tool_calls
    current_vision_text: str   # Node A 输出的纯文本场景描述
    healthy: bool              # 当前行为是否健康
    timestamp: str             # 本轮感知触发时间


# ── Node A：本地边缘感知层（在 graph.py 中作为包装器存在） ────
def perception_node(state: AgentState) -> dict:
    """
    Node A 包装器：在 LangGraph 流转中被调用。
    真实的摄像头 + MediaPipe + Moondream2 逻辑在 perception.py 中。
    当由 main.py 触发时，此节点接收已由感知线程写入的 vision_text。
    """
    vision_text = state.get("current_vision_text", "")
    timestamp = state.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    console.print(f"{LOG_PERCEPTION} t={timestamp} vision=\"{vision_text}\"")

    # 判定是否健康（简单关键词过滤，后续可接小模型分类）
    unhealthy_keywords = [
        "躺", "刷手机", "玩游戏", "睡觉", "看视频", "发呆",
        "lying", "phone", "gaming", "sleeping", "watching video", "idle",
    ]
    is_unhealthy = any(kw in vision_text.lower() for kw in unhealthy_keywords)
    healthy = not is_unhealthy

    route = "healthy -> END" if healthy else "unhealthy -> [B]"
    console.print(f"{LOG_PERCEPTION} {route}")

    return {
        "healthy": healthy,
        "timestamp": timestamp,
        "messages": [
            HumanMessage(
                content=(
                    f"[系统时间: {timestamp}]\n"
                    f"本地感知层报告：{vision_text}\n"
                    f"请根据宿主当前行为判断是否需要惩罚。"
                )
            )
        ],
    }


# ── Node B：云端中枢决策层（DeepSeek，当前为 STUB） ──────────
def decision_node(state: AgentState) -> dict:
    """
    Node B：读取感知层纯文本，调用 DeepSeek LLM 判决是否惩罚。
    当前为 MOCK 实现，返回模拟决策结果。
    TODO: 替换为真实 DeepSeek API 调用 + function calling。
    """
    vision_text = state.get("current_vision_text", "")
    console.print(f"{LOG_DECISION} [MOCK] deepseek evaluating: \"{vision_text}\"")
    mock_response = AIMessage(content="[MOCK] violation detected, executing punishment")
    console.print(f"{LOG_DECISION} verdict=punish -> [C]")

    return {"messages": [mock_response]}


# ── Node C：本地物理/数字执行层 ──────────────────────────────
def execution_node(state: AgentState) -> dict:
    """
    Node C：拦截 LLM tool_calls 并在本地执行。
    当前为 MOCK 实现。
    TODO: 接入真实 LangGraph ToolNode。
    """
    console.print(f"{LOG_EXECUTION} [MOCK] tool=play_tts_punishment")
    return {
        "messages": [
            AIMessage(content="[MOCK] 惩罚已执行完毕，本轮循环结束。")
        ]
    }


# ── 路由逻辑 ─────────────────────────────────────────────────
def route_after_perception(state: AgentState) -> str:
    """Node A → 健康则 END，不健康则进入 Node B"""
    return END if state.get("healthy", True) else "decision"


def route_after_decision(state: AgentState) -> str:
    """Node B → 无 tool_calls 则 END，有则进入 Node C"""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execution"
    return END


# ── 构建 LangGraph 图 ─────────────────────────────────────────
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("perception", perception_node)
    builder.add_node("decision", decision_node)
    builder.add_node("execution", execution_node)

    builder.add_edge(START, "perception")
    builder.add_conditional_edges(
        "perception",
        route_after_perception,
        {"decision": "decision", END: END},
    )
    builder.add_conditional_edges(
        "decision",
        route_after_decision,
        {"execution": "execution", END: END},
    )
    builder.add_edge("execution", END)

    return builder.compile()
