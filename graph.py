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
        # parallel_tool_calls=False：强制 LLM 每步只调用一批工具，
        # 保证 observe_camera 不会与其他工具混发
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

## 工具调用规则（严格遵守）
- **惩罚工具**（play_tts_punishment / send_wechat_shame_message / force_close_app / cut_smart_plug_power）：可以在同一步骤里同时调用多个。
- **observe_camera**：必须单独调用，绝对不能和其他任何工具同时调用。它需要等待宿主响应，混用会导致观察无效。

## 标准惩罚流程（按顺序执行）
1. **初次发现摆烂** → 单独调用 play_tts_punishment 发出语音警告，骂得狠一点。
2. **警告后** → 单独调用 observe_camera，等待宿主反应。
3. **还在摆烂** → 同时调用多个惩罚工具升级打击（play_tts_punishment + send_wechat_shame_message，或其他组合）。
4. **已经收手** → 冷嘲热讽一句就结束，不再调用工具。

## 内容风格
- 极度口语化，像微信骂人，短平快，绝不超过两三句（50字以内）。
- 必须使用"妈的"、"操"、"傻逼"、"废物"等词汇。
- 禁止条列式排版，禁止共情。

## 自律行为处理
如果一开始就在自律（学习/工作/锻炼/睡觉），冷嘲一句收手，不调用任何工具。"""


def _reorder_and_repair(messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[ToolMessage]]:
    """
    重建消息序列，确保每条 AIMessage(tool_calls) 后紧跟对应的 ToolMessage。
    处理两种情况：
      1. ToolMessage 存在但顺序错误（在后续消息中）→ 取出后内联插入
      2. ToolMessage 完全缺失（孤悬 tool_call）      → 生成占位 ToolMessage

    返回 (repaired_sequence, new_repairs)：
      repaired_sequence — 顺序正确、可直接发给 LLM 的消息列表
      new_repairs       — 本次新生成的占位 ToolMessage（需持久化到 checkpoint）
    使用确定性 ID（repair_{tc_id}）保证幂等，同一孤悬不会重复修复。
    """
    # 建立 tool_call_id → ToolMessage 的查找表
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
            continue  # 统一在 AIMessage 之后内联放置，跳过原位

        result.append(msg)

        if not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)):
            continue

        # 紧跟 AIMessage 放置每个 tool_call 的响应
        for tc in msg.tool_calls:
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if not tc_id or tc_id in placed_ids:
                continue

            if tc_id in tool_response_map:
                result.append(tool_response_map[tc_id])          # 已有，内联
            else:
                repair = ToolMessage(
                    content="[aborted: interrupted by max_iterations limit]",
                    tool_call_id=tc_id,
                    id=f"repair_{tc_id}",  # 确定性 ID，幂等
                )
                result.append(repair)
                new_repairs.append(repair)
                console.print(f"{LOG_DECISION} repair orphaned tool_call id={tc_id[:12]}...")

            placed_ids.add(tc_id)

    return result, new_repairs


def decision_node(state: AgentState) -> dict:
    iteration = state.get("react_iterations", 0)

    # ── 提前拦截 max_iterations ────────────────────────────────
    # 必须在 LLM 调用之前检查，防止产生新的孤悬 AIMessage(tool_calls)
    if iteration >= REACT_MAX_ITERATIONS:
        console.print(f"{LOG_DECISION} max iterations ({REACT_MAX_ITERATIONS}) reached, ending gracefully")
        return {"react_iterations": 0}

    console.print(f"{LOG_DECISION} calling DeepSeek... [react iter={iteration}]")

    raw_messages = state.get("messages", [])

    # trim_messages：只保留最近 N 条历史，防止 context 爆炸
    trimmed = trim_messages(
        raw_messages,
        strategy="last",
        token_counter=len,
        max_tokens=CONTEXT_MAX_MESSAGES,
        start_on="human",
        end_on=("human", "tool"),
        include_system=False,
    )

    # ── 重建序列：修复顺序 + 补全缺失的 ToolMessage ───────────
    # 必须在 trim 之后做，确保发给 LLM 的序列顺序绝对正确
    trimmed, new_repairs = _reorder_and_repair(trimmed)

    # 若有压缩摘要，追加到 system prompt（语义上属于系统记忆，非用户输入）
    summary = state.get("conversation_summary", "")
    system_content = _SYSTEM_PROMPT
    if summary:
        system_content += f"\n\n[历史摘要] {summary}"

    messages = [SystemMessage(content=system_content)] + trimmed

    llm = _get_llm()
    try:
        response = llm.invoke(messages)
    except Exception as e:
        console.print(f"{LOG_DECISION} LLM error: {e}")
        return {"react_iterations": 0}  # H1: 降级结束本轮，不崩溃

    # new_repairs 持久化到 checkpoint（确定性 ID 保证幂等），再追加 LLM response
    updates: dict = {"messages": new_repairs + [response]}
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
            # 惩罚流程结束时把最终结语播报出来（iteration>0 说明本轮有过惩罚动作）
            if iteration > 0:
                from tools import play_tts_punishment
                try:
                    play_tts_punishment.invoke({"text": response.content})
                except Exception:
                    pass
        updates["react_iterations"] = 0  # 本轮结束，重置
        if iteration == 0:
            # 全程未调用工具，说明宿主健康
            updates["consecutive_healthy"] = state.get("consecutive_healthy", 0) + 1

    # 触发摘要压缩：只在 iter=0（新一轮观察开始前）执行，避免 ReAct 中途打断
    # 用列表合并而非 dict.update，防止覆盖已写入的 response/repairs
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
    # 只看最后一条消息是否带 tool_calls：
    #   - 有 → 去 execution（始终执行，避免孤悬 tool_call 污染 checkpoint）
    #   - 无 → END（decision_node 已在内部处理 max_iterations 早返回）
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
