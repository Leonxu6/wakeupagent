"""
graph.py — LangGraph 状态机定义 v3
架构：daily_reset → perception → decision ⇌ execution(ToolNode, ReAct loop)
特性：
  - SqliteSaver 持久化 checkpointer（跨进程状态保存）
  - qwen2.5 小脑替代 Planner，动态路由决策
  - trim_messages 防止 context 爆炸，summarize 做记忆压缩
  - ToolNode 原生 ReAct 循环（execution → decision），最多 4 轮
  - 每日自动清空 + 生成日报存档
"""
from __future__ import annotations

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Annotated

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage,
    ToolMessage, trim_messages,
)
from langchain_openai import ChatOpenAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from typing_extensions import TypedDict, NotRequired

from config import (
    LOG_A, LOG_B, LOG_C,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
    CHECKPOINT_DB_PATH, DAILY_REPORT_PATH,
    CONTEXT_MAX_MESSAGES, SUMMARIZE_THRESHOLD, REACT_MAX_ITERATIONS,
)
from tools import ALL_TOOLS

LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
LOG_RESET      = "[blue][R][/blue]"

console = Console()
_ERROR_DETAIL_LIMIT = 120


def _safe_error_detail(exc: object) -> str:
    """Return a bounded error class label without backend message contents."""
    name = exc.__class__.__name__ if isinstance(exc, BaseException) else "Error"
    return name[:_ERROR_DETAIL_LIMIT]


# ── 全局状态定义 ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_vision_text: NotRequired[str]
    healthy:             NotRequired[bool]
    timestamp:           NotRequired[str]
    should_escalate:     NotRequired[bool]
    react_iterations:    NotRequired[int]
    session_date:        NotRequired[str]
    unhealthy_count:     NotRequired[int]
    consecutive_healthy: NotRequired[int]
    conversation_summary: NotRequired[str]


_llm_with_tools = None

def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.7,
            timeout=30,
            max_retries=1,
        )
        _llm_with_tools = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    return _llm_with_tools


def _get_llm_plain():
    """不带工具的 LLM，用于摘要/日报生成"""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        timeout=30,
        max_retries=1,
    )


def daily_reset_node(state: AgentState) -> dict:
    """检测日期变更并生成昨日摘要，随后重置每日计数。"""
    today = date.today().isoformat()
    session_date = state.get("session_date", "")
    if not session_date:
        return {
            "session_date": today,
            "unhealthy_count": 0,
            "consecutive_healthy": 0,
            "react_iterations": 0,
        }
    if session_date == today:
        return {}

    messages = state.get("messages", [])
    console.print(f"{LOG_RESET} new day detected ({session_date} → {today}), generating report...")
    report_text = _generate_daily_report(messages, session_date, state)
    _save_daily_report(report_text, session_date)
    console.print(f"{LOG_RESET} report saved → {DAILY_REPORT_PATH}")
    delete_ops = [RemoveMessage(id=m.id) for m in messages if hasattr(m, 'id') and m.id]
    return {
        "messages": delete_ops,
        "session_date": today,
        "unhealthy_count": 0,
        "consecutive_healthy": 0,
        "react_iterations": 0,
        "conversation_summary": report_text,
    }


def _generate_daily_report(messages: list, date_str: str, state: AgentState) -> str:
    if not messages and not date_str:
        return ""
    unhealthy = state.get("unhealthy_count", 0)
    summary_prompt = (
        f"请用50字以内总结 {date_str} 的专注情况。"
        f"今日共检测到 {unhealthy} 次需要重新聚焦的时刻。"
        f"给出中性、具体、可执行的明日建议。"
    )
    try:
        llm = _get_llm_plain()
        context = messages[-10:] if len(messages) > 10 else messages
        response = llm.invoke(context + [HumanMessage(content=summary_prompt)])
        return response.content
    except Exception as exc:  # noqa: BLE001
        console.print(f"{LOG_RESET} report generation failed ({_safe_error_detail(exc)})")
        return f"{date_str}: 检测到 {unhealthy} 次需要重新聚焦的时刻，报告生成失败。"


def _save_daily_report(report: str, date_str: str):
    path = Path(DAILY_REPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {date_str}\n{report}\n")


def perception_node(state: AgentState) -> dict:
    vision_text = state.get("current_vision_text", "")
    timestamp = state.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    should_escalate = state.get("should_escalate", False)
    console.print(f"{LOG_PERCEPTION} t={timestamp} vision=\"{vision_text[:80]}\"")
    route = "escalate -> [B]" if should_escalate else "pass -> END"
    console.print(f"{LOG_PERCEPTION} cerebellum decision: {route}")
    parts = [
        f"[系统时间: {timestamp}]",
        f"【摄像头报告】{vision_text}",
        "根据宿主当前行为判断是否需要温和提醒或经用户明确启用的辅助动作。",
    ]
    return {"timestamp": timestamp, "messages": [HumanMessage(content="\n".join(parts))]}


_SYSTEM_PROMPT = """你是一个简短、明确的专注监督助手。目标是帮助用户重新聚焦，而不是羞辱、威胁或制造破坏。

## 多轮行动原则
你可以连续行动多次，但优先使用最小干预。每次工具执行后根据结果决定是否还需要下一步。

## 工具调用规则
- play_tts_punishment 可以用于简短的语音提醒，但内容应直接、克制、不辱骂用户。
- send_wechat_shame_message、open_webpage、force_close_app 都属于明显副作用能力。只有工具本身已通过本地配置显式启用时才可能执行；不要尝试绕过工具返回的 disabled/error 状态。
- observe_camera 必须单独调用，不能与其他工具同时调用。
- legacy chaos mode 已移除且不得请求、描述或模拟。不要尝试通过其他工具组合复现其效果。

## 推荐流程
1. 初次发现偏离计划：给出一句具体提醒，必要时单独调用语音提醒。
2. 需要确认状态：单独调用 observe_camera，再根据新观察判断。
3. 若用户已明确启用某个副作用能力，可在必要时选择一个最小影响的动作；不要一次堆叠多个干扰动作。
4. 一旦用户回到计划中的活动，停止工具调用。

## 内容风格
- 口语化、简短，通常不超过两三句。
- 聚焦当前行为和下一步动作，不进行人身攻击、羞辱、威胁或贬低。
- 每次调用工具时在 content 中说明为什么需要该动作。
- 工具失败或被禁用时接受该结果，不重复强行调用。

## 自律行为处理
如果一开始就在学习、工作、锻炼、休息或其他用户计划中的活动，简短确认后结束，不调用任何工具。"""


def _reorder_and_repair(messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[ToolMessage]]:
    tool_response_map: dict[str, ToolMessage] = {
        m.tool_call_id: m
        for m in messages
        if isinstance(m, ToolMessage) and hasattr(m, "tool_call_id")
    }
    result: list[BaseMessage] = []
    new_repairs: list[ToolMessage] = []
    placed_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        result.append(msg)
        if not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)):
            continue
        for tc in msg.tool_calls:
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if not tc_id or tc_id in placed_ids:
                continue
            if tc_id in tool_response_map:
                result.append(tool_response_map[tc_id])
            else:
                repair = ToolMessage(
                    content="[aborted: interrupted by max_iterations limit]",
                    tool_call_id=tc_id,
                    id=f"repair_{tc_id}",
                )
                result.append(repair)
                new_repairs.append(repair)
                console.print(f"{LOG_DECISION} repair orphaned tool_call id={tc_id[:12]}...")
            placed_ids.add(tc_id)
    return result, new_repairs


def decision_node(state: AgentState) -> dict:
    iteration = state.get("react_iterations", 0)
    if iteration >= REACT_MAX_ITERATIONS:
        console.print(f"{LOG_DECISION} max iterations ({REACT_MAX_ITERATIONS}) reached, ending gracefully")
        return {"react_iterations": 0}
    console.print(f"{LOG_DECISION} calling DeepSeek... [react iter={iteration}]")
    raw_messages = state.get("messages", [])
    trimmed = trim_messages(
        raw_messages,
        strategy="last",
        token_counter=len,
        max_tokens=CONTEXT_MAX_MESSAGES,
        start_on="human",
        end_on=("human", "tool"),
        include_system=False,
    )
    trimmed, new_repairs = _reorder_and_repair(trimmed)
    summary = state.get("conversation_summary", "")
    system_content = _SYSTEM_PROMPT
    if summary:
        system_content += f"\n\n[历史摘要] {summary}"
    messages = [SystemMessage(content=system_content)] + trimmed
    llm = _get_llm()
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        console.print(f"{LOG_DECISION} LLM error ({_safe_error_detail(exc)})")
        return {"react_iterations": 0}
    updates: dict = {"messages": new_repairs + [response]}
    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        console.print(f"{LOG_DECISION} tools={tool_names} -> [C] (iter {iteration+1})")
        if response.content and "play_tts_punishment" not in tool_names:
            console.print(f"{LOG_DECISION} ▶ {response.content[:100]}")
            from tools import play_tts_punishment
            try:
                play_tts_punishment.invoke({"text": response.content})
            except Exception:  # noqa: BLE001
                pass
        updates["react_iterations"] = iteration + 1
        if iteration == 0:
            updates["unhealthy_count"] = state.get("unhealthy_count", 0) + 1
            updates["consecutive_healthy"] = 0
    else:
        console.print(f"{LOG_DECISION} verdict=done -> END (total iters={iteration})")
        if response.content:
            console.print(f"{LOG_DECISION} {response.content[:100]}")
            from tools import play_tts_punishment
            try:
                play_tts_punishment.invoke({"text": response.content})
            except Exception:  # noqa: BLE001
                pass
        updates["react_iterations"] = 0
        if iteration == 0:
            updates["consecutive_healthy"] = state.get("consecutive_healthy", 0) + 1
    if len(raw_messages) >= SUMMARIZE_THRESHOLD and iteration == 0:
        summarize_result = _summarize_messages(raw_messages, state)
        updates["messages"] = updates.get("messages", []) + summarize_result.pop("messages", [])
        updates.update(summarize_result)
    return updates


def _summarize_messages(messages: list, state: AgentState) -> dict:
    """将历史消息压缩为摘要，删除旧消息，保留最新5条上下文。"""
    console.print(f"{LOG_DECISION} summarizing {len(messages)} messages...")
    summary_so_far = state.get("conversation_summary", "")
    prefix = f"已有摘要：{summary_so_far}\n\n请在此基础上更新：" if summary_so_far else "请总结以下对话："
    try:
        llm = _get_llm_plain()
        response = llm.invoke(
            messages[-20:] + [HumanMessage(content=prefix + "（50字以内，记录关键偏离与重新聚焦动作）")]
        )
        new_summary = response.content
    except Exception:  # noqa: BLE001
        new_summary = summary_so_far
    to_delete = messages[:-5]
    delete_ops = [RemoveMessage(id=m.id) for m in to_delete if hasattr(m, 'id') and m.id]
    return {"conversation_summary": new_summary, "messages": delete_ops}


def route_after_perception(state: AgentState) -> str:
    return "decision" if state.get("should_escalate", False) else END


def route_after_decision(state: AgentState) -> str:
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execution"
    return END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("daily_reset", daily_reset_node)
    builder.add_node("perception",  perception_node)
    builder.add_node("decision",    decision_node)
    builder.add_node("execution",   ToolNode(ALL_TOOLS))
    builder.add_edge(START,         "daily_reset")
    builder.add_edge("daily_reset", "perception")
    builder.add_conditional_edges(
        "perception", route_after_perception,
        {"decision": "decision", END: END},
    )
    builder.add_conditional_edges(
        "decision", route_after_decision,
        {"execution": "execution", END: END},
    )
    builder.add_edge("execution", "decision")
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    return builder.compile(checkpointer=checkpointer)
