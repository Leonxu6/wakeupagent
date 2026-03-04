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
    trim_messages,
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
    CONTEXT_MAX_MESSAGES, SUMMARIZE_THRESHOLD,
)
from tools import ALL_TOOLS

LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
LOG_RESET      = "[blue][R][/blue]"

console = Console()


# ── 全局状态定义 ──────────────────────────────────────────────
class AgentState(TypedDict):
    # 核心对话历史（add_messages 自动追加，配合 checkpointer 跨轮保留）
    messages: Annotated[list[BaseMessage], add_messages]

    # 感知层（每轮由感知回调注入，非持久字段）
    current_vision_text: NotRequired[str]
    healthy:             NotRequired[bool]
    timestamp:           NotRequired[str]

    # 小脑路由决策
    should_escalate:     NotRequired[bool]   # 小脑决定是否上报大脑

    # ReAct 循环控制（每轮结束重置为 0）
    react_iterations:    NotRequired[int]

    # 跨轮统计（checkpointer 持久化）
    session_date:        NotRequired[str]    # 当天日期，用于每日清空判断
    unhealthy_count:     NotRequired[int]    # 今日摆烂次数
    consecutive_healthy: NotRequired[int]    # 连续健康次数

    # 记忆压缩
    conversation_summary: NotRequired[str]


# ── LLM 单例 ─────────────────────────────────────────────────
_llm_with_tools = None

def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.7,
            timeout=30,      # H3: 防止 API 卡死阻塞监控循环
            max_retries=1,
        )
        # H4: 关闭并行调用，确保 observe_camera 在惩罚工具之后顺序执行
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


# ── Node R：每日清空 + 日报归档 ──────────────────────────────
def daily_reset_node(state: AgentState) -> dict:
    """
    检测日期变更 → 生成昨日报告存档 → 清空消息历史 → 重置计数器。
    每轮图执行的入口，无日期变更时 0 开销直接透传。
    """
    today = date.today().isoformat()
    session_date = state.get("session_date", "")

    if session_date == today:
        return {}  # 同一天，无操作

    messages = state.get("messages", [])
    console.print(f"{LOG_RESET} new day detected ({session_date} → {today}), generating report...")

    # 生成昨日日报
    report_text = _generate_daily_report(messages, session_date, state)

    # 归档到文件
    _save_daily_report(report_text, session_date)
    console.print(f"{LOG_RESET} report saved → {DAILY_REPORT_PATH}")

    # 清空所有旧消息
    delete_ops = [RemoveMessage(id=m.id) for m in messages if hasattr(m, 'id') and m.id]

    return {
        "messages": delete_ops,
        "session_date": today,
        "unhealthy_count": 0,
        "consecutive_healthy": 0,
        "react_iterations": 0,              # M1: 防止崩溃重启后遗留中间状态
        "conversation_summary": report_text, # 日报作为新一天的初始摘要
    }


def _generate_daily_report(messages: list, date_str: str, state: AgentState) -> str:
    if not messages and not date_str:
        return ""
    unhealthy = state.get("unhealthy_count", 0)
    summary_prompt = (
        f"请用50字以内总结 {date_str} 的自律监督情况。"
        f"今日共摆烂 {unhealthy} 次。"
        f"给出整体评价和明日建议。"
    )
    try:
        llm = _get_llm_plain()
        context = messages[-10:] if len(messages) > 10 else messages
        response = llm.invoke(context + [HumanMessage(content=summary_prompt)])
        return response.content
    except Exception as e:
        console.print(f"{LOG_RESET} report generation failed: {e}")
        return f"{date_str}: 摆烂 {unhealthy} 次，报告生成失败。"


def _save_daily_report(report: str, date_str: str):
    path = Path(DAILY_REPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {date_str}\n{report}\n")


# ── Node A：感知层包装器 ──────────────────────────────────────
def perception_node(state: AgentState) -> dict:
    vision_text = state.get("current_vision_text", "")
    timestamp = state.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    should_escalate = state.get("should_escalate", False)

    console.print(f"{LOG_PERCEPTION} t={timestamp} vision=\"{vision_text[:80]}\"")
    route = "escalate -> [B]" if should_escalate else "pass -> END"
    console.print(f"{LOG_PERCEPTION} cerebellum decision: {route}")

    return {
        "timestamp": timestamp,
        "messages": [
            HumanMessage(content=(
                f"[系统时间: {timestamp}]\n"
                f"本地感知层报告：{vision_text}\n"
                f"请根据宿主当前行为判断是否需要惩罚。"
            ))
        ],
    }


# ── Node B：决策层（DeepSeek + parallel function calling） ────
_SYSTEM_PROMPT = """你是一个极其暴躁、满嘴脏话的真人监工。所有"AI助手"、"礼貌"、"排版"的设定全他妈删了。

## 你的多轮行动能力
你可以连续行动多次。每次行动后你会收到工具的执行结果，根据结果决定下一步。
善用这个能力做【渐进式惩罚】，而不是一上来就把所有工具全打出去。

## 标准惩罚流程（按顺序执行）
1. **初次发现摆烂** → 只调用 play_tts_punishment 发出语音警告，骂得狠一点。
2. **警告后** → 必须调用 observe_camera 重新看一眼，确认他有没有收手。
3. **还在摆烂** → 升级：play_tts_punishment（更狠）+ 其他惩罚工具组合（force_close_app / send_wechat_shame_message / cut_smart_plug_power）。
4. **已经收手** → 冷嘲热讽一句就结束，不再调用工具。

## 内容风格
- 极度口语化，像微信骂人，短平快，绝不超过两三句（50字以内）。
- 必须使用"妈的"、"操"、"傻逼"、"废物"等词汇。
- 禁止条列式排版，禁止共情。

## 自律行为处理
如果一开始就在自律（学习/工作/锻炼/睡觉），冷嘲一句收手，不调用任何工具。"""


REACT_MAX_ITERATIONS = 4  # 最多 4 轮 decision↔execution，防止死循环


def decision_node(state: AgentState) -> dict:
    iteration = state.get("react_iterations", 0)
    console.print(f"{LOG_DECISION} calling DeepSeek... [react iter={iteration}]")

    # trim_messages：只保留最近 N 条历史，防止 context 爆炸
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

    # 若有压缩摘要，注入为第一条 human 消息
    summary = state.get("conversation_summary", "")
    prefix = [HumanMessage(content=f"[历史摘要] {summary}")] if summary else []

    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + prefix + trimmed

    llm = _get_llm()
    try:
        response = llm.invoke(messages)
    except Exception as e:
        console.print(f"{LOG_DECISION} LLM error: {e}")
        return {"react_iterations": 0}  # H1: 降级结束本轮，不崩溃

    updates: dict = {"messages": [response]}
    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        console.print(f"{LOG_DECISION} tools={tool_names} -> [C] (iter {iteration+1})")
        updates["react_iterations"] = iteration + 1
        # unhealthy_count 只在第一轮计入（避免同一事件重复计数）
        if iteration == 0:
            updates["unhealthy_count"] = state.get("unhealthy_count", 0) + 1
            updates["consecutive_healthy"] = 0
    else:
        console.print(f"{LOG_DECISION} verdict=done -> END (total iters={iteration})")
        if response.content:
            console.print(f"{LOG_DECISION} {response.content[:100]}")
        updates["react_iterations"] = 0  # 本轮结束，重置
        if iteration == 0:
            # 全程未调用工具，说明宿主健康
            updates["consecutive_healthy"] = state.get("consecutive_healthy", 0) + 1

    # 触发摘要压缩
    if len(raw_messages) >= SUMMARIZE_THRESHOLD:
        updates.update(_summarize_messages(raw_messages, state))

    return updates


def _summarize_messages(messages: list, state: AgentState) -> dict:
    """将历史消息压缩为摘要，删除旧消息，保留最新5条上下文。"""
    console.print(f"{LOG_DECISION} summarizing {len(messages)} messages...")
    summary_so_far = state.get("conversation_summary", "")
    prefix = f"已有摘要：{summary_so_far}\n\n请在此基础上更新：" if summary_so_far else "请总结以下对话："
    try:
        llm = _get_llm_plain()
        response = llm.invoke(
            messages[-20:] + [HumanMessage(content=prefix + "（50字以内，记录关键违规与惩罚）")]
        )
        new_summary = response.content
    except Exception:
        new_summary = summary_so_far

    # 删除旧消息，保留最新5条
    to_delete = messages[:-5]
    delete_ops = [RemoveMessage(id=m.id) for m in to_delete if hasattr(m, 'id') and m.id]
    return {"conversation_summary": new_summary, "messages": delete_ops}


# ── 路由逻辑 ─────────────────────────────────────────────────
def route_after_perception(state: AgentState) -> str:
    return "decision" if state.get("should_escalate", False) else END


def route_after_decision(state: AgentState) -> str:
    if state.get("react_iterations", 0) >= REACT_MAX_ITERATIONS:
        console.print(f"{LOG_DECISION} max iterations reached -> END")
        return END
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execution"
    return END


# ── 构建图 ───────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("daily_reset", daily_reset_node)
    builder.add_node("perception",  perception_node)
    builder.add_node("decision",    decision_node)
    builder.add_node("execution",   ToolNode(ALL_TOOLS))  # 原生并行执行

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
    builder.add_edge("execution", "decision")  # ReAct 循环

    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    return builder.compile(checkpointer=checkpointer)
