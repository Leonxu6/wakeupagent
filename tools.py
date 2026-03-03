"""
tools.py — Node C 本地物理/数字执行工具库
每个工具必须带 @tool 装饰器和详尽的 Docstring（LLM 依赖 Docstring 选工具）。

当前状态：Mock 实现，用于验证图流转逻辑。
后续替换为真实执行（PyAutoGUI, TTS, IoT 等）。
"""
from langchain_core.tools import tool
from rich.console import Console

console = Console()


@tool
def send_wechat_shame_message(target_contact: str, message: str) -> str:
    """
    强制接管鼠标键盘，打开微信，向指定联系人发送一条"社死"消息。
    当宿主需要被社交羞辱惩罚时使用此工具。

    Args:
        target_contact: 微信联系人姓名（如"妈妈"、"班主任"）
        message: 要发送的羞辱性消息内容

    Returns:
        执行结果描述
    """
    console.print(
        f"[bold red]🦾 [本地执行] MOCK — 模拟接管鼠标，打开微信\n"
        f"   联系人: {target_contact}\n"
        f"   消息: {message}[/bold red]"
    )
    return f"[MOCK] 已向 {target_contact} 发送消息：{message}"


@tool
def play_tts_punishment(text: str) -> str:
    """
    通过 TTS 语音通道朗读 LLM 生成的嘲讽语音，在扬声器中播放。
    当需要用声音惩罚宿主时使用此工具。

    Args:
        text: 要朗读的嘲讽文本（由 LLM 生成）

    Returns:
        执行结果描述
    """
    console.print(
        f"[bold red]🔊 [本地执行] MOCK — 模拟 TTS 播报\n"
        f"   内容: {text}[/bold red]"
    )
    return f"[MOCK] TTS 已播放：{text}"


@tool
def cut_smart_plug_power(device_name: str) -> str:
    """
    通过局域网 HTTP 请求，切断指定智能插座的电源，强制关闭宿主正在使用的设备。
    当宿主沉迷电子设备、拒绝停止时使用此工具。

    Args:
        device_name: 智能插座设备名（如"电脑插座"、"游戏主机"）

    Returns:
        执行结果描述
    """
    console.print(
        f"[bold red]⚡ [本地执行] MOCK — 模拟发送 IoT 断电指令\n"
        f"   设备: {device_name}[/bold red]"
    )
    return f"[MOCK] 已切断 {device_name} 电源"


@tool
def force_close_app(app_name: str) -> str:
    """
    强制关闭宿主正在运行的指定应用程序窗口（如游戏、视频播放器）。
    当宿主在应该学习时玩游戏或看视频时使用此工具。

    Args:
        app_name: 应用程序名称（如"Steam"、"Netflix"、"YouTube"）

    Returns:
        执行结果描述
    """
    console.print(
        f"[bold red]💀 [本地执行] MOCK — 模拟强制关闭应用\n"
        f"   应用: {app_name}[/bold red]"
    )
    return f"[MOCK] 已强制关闭 {app_name}"


# 工具列表（供 LangGraph ToolNode 注册）
ALL_TOOLS = [
    send_wechat_shame_message,
    play_tts_punishment,
    cut_smart_plug_power,
    force_close_app,
]
