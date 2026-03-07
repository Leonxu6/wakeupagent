"""
tools.py — Node C 本地物理/数字执行工具库
每个工具必须带 @tool 装饰器和详尽的 Docstring（LLM 依赖 Docstring 选工具）。

已实现：play_tts_punishment / send_wechat_shame_message / open_webpage /
        force_close_app / chaos_terminal_punishment / observe_camera
"""
import os
import subprocess
import tempfile
import time
import webbrowser

from langchain_core.tools import tool
from rich.console import Console
from config import CAPTURE_INTERVAL_SEC

console = Console()

# macOS TTS 普通话语音（系统偏好 > 辅助功能 > 朗读内容 可安装更多声音）
_TTS_VOICE = "Tingting"


@tool
def play_tts_punishment(text: str) -> str:
    """
    通过 macOS 系统 TTS 朗读嘲讽语音，从扬声器播出，让宿主听见惩罚。
    当需要用声音警告/辱骂宿主时使用此工具。播放期间阻塞，完成后返回。

    Args:
        text: 要朗读的嘲讽文本（由 LLM 生成，50字以内效果最佳）

    Returns:
        执行结果描述
    """
    console.print(f"[bold red]🔊 [TTS] {text}[/bold red]")
    try:
        subprocess.run(["say", "-v", _TTS_VOICE, text], timeout=60, check=True)
        return f"TTS 播放完毕：{text}"
    except FileNotFoundError:
        return "Error: say 命令不存在（仅支持 macOS）"
    except subprocess.TimeoutExpired:
        return "Error: TTS 播放超时"
    except subprocess.CalledProcessError as e:
        # 声音不存在时降级到系统默认
        try:
            subprocess.run(["say", text], timeout=60, check=True)
            return f"TTS 播放完毕（默认声音）：{text}"
        except Exception as e2:
            return f"Error: {e2}"
    except Exception as e:
        return f"Error: {e}"


@tool
def send_wechat_shame_message(target: str, message: str) -> str:
    """
    通过 osascript System Events 驱动微信 Mac 客户端，向固定联系人发送羞辱消息。
    不依赖窗口焦点，直接向微信进程发送键盘事件，更可靠。
    需要在系统设置 → 隐私与安全性 → 辅助功能 中授权终端（Terminal/iTerm2）。

    target 只能是以下之一（在 config.py WECHAT_CONTACTS 中配置真实名字）：
    - "老妈"   → 向妈妈发消息，用来让她来管你
    - "导师"   → 向导师发消息，增加学术压力
    - "班级群" → 在班级群里公开社死

    Args:
        target: 接收目标，必须是 "老妈" / "导师" / "班级群" 之一
        message: 要发送的羞辱/催促内容，由 LLM 生成

    Returns:
        执行结果描述
    """
    from config import WECHAT_CONTACTS
    contact = WECHAT_CONTACTS.get(target)
    if not contact:
        return f"Error: 不支持的 target '{target}'，只能用: {list(WECHAT_CONTACTS.keys())}"

    console.print(f"[bold red]🦾 [WeChat] → {target}({contact}): {message}[/bold red]")

    def _esc(s: str) -> str:
        """转义字符串供 AppleScript 双引号字符串使用"""
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # open -a WeChat 能真正把 WeChat 带到前台（tell application activate 无效）
    script = f'''
do shell script "open -a WeChat"
delay 2.0
tell application "System Events"
    tell process "WeChat"
        keystroke "f" using {{command down}}
        delay 1.0
        keystroke "a" using {{command down}}
        set the clipboard to "{_esc(contact)}"
        keystroke "v" using {{command down}}
        delay 2.0
        keystroke return
        delay 1.0
        set the clipboard to "{_esc(message)}"
        keystroke "v" using {{command down}}
        delay 0.5
        keystroke return
    end tell
end tell
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip() or 'osascript 执行失败'}"
        return f"已向 {target}({contact}) 发送：{message}"
    except subprocess.TimeoutExpired:
        return "Error: 微信操作超时"
    except Exception as e:
        return f"Error: {e}"


@tool
def open_webpage(url: str) -> str:
    """
    在默认浏览器中强制打开一个网页，用于惩罚/骚扰宿主。
    当宿主在玩游戏/摸鱼时，打开学习网页或抽象内容干扰其娱乐。

    适合的 URL 类型：
    - 强制学习：https://leetcode.cn（强制刷题）
    - 学习视频：https://www.bilibili.com/search?keyword=高数 （搜索枯燥课程）
    - 摸鱼克星：https://arxiv.org （满屏论文）
    - 其他任何能让宿主感到不适的内容

    Args:
        url: 要打开的完整 URL（必须以 http:// 或 https:// 开头）

    Returns:
        执行结果描述
    """
    console.print(f"[bold red]🌐 [浏览器] 强制打开 {url}[/bold red]")
    try:
        webbrowser.open(url)
        return f"已在浏览器中打开：{url}"
    except Exception as e:
        return f"Error: {e}"


@tool
def force_close_app(app_name: str) -> str:
    """
    强制关闭宿主正在运行的指定应用程序（如游戏、视频播放器、摸鱼软件）。
    当宿主在应该学习时玩游戏或看视频时使用此工具。
    先尝试 osascript 优雅退出，失败则 killall，再失败则 pkill -i 模糊匹配。

    Args:
        app_name: 应用程序名称（如"Steam"、"Bilibili"、"WeChat"、"Safari"）

    Returns:
        执行结果描述
    """
    console.print(f"[bold red]💀 [force_close] 关闭应用: {app_name}[/bold red]")
    errors = []

    # 1. osascript 优雅退出
    try:
        r = subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to quit'],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return f"已通过 osascript 关闭 {app_name}"
        errors.append(f"osascript: {r.stderr.strip()}")
    except Exception as e:
        errors.append(f"osascript: {e}")

    # 2. killall 精确匹配
    try:
        r = subprocess.run(["killall", app_name], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return f"已通过 killall 关闭 {app_name}"
        errors.append(f"killall: {r.stderr.strip()}")
    except Exception as e:
        errors.append(f"killall: {e}")

    # 3. pkill -i 模糊匹配
    try:
        r = subprocess.run(["pkill", "-i", app_name], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return f"已通过 pkill 关闭 {app_name}"
        errors.append(f"pkill: {r.stderr.strip()}")
    except Exception as e:
        errors.append(f"pkill: {e}")

    return f"Error: 无法关闭 {app_name}。详情: {'; '.join(errors)}"


@tool
def chaos_terminal_punishment(message: str) -> str:
    """
    Python 脚本狂暴模式：瞬间打开 50 个 Terminal 窗口，每个窗口用绿色字体疯狂打印
    宿主的罪状或《出师表》10000 遍，同时启动 5 种语言 TTS 并发朗读，彻底摧毁宿主的
    摸鱼环境。当宿主屡教不改、需要终极惩罚时使用此工具。

    Args:
        message: 要在终端中打印和朗读的惩罚文字（由 LLM 生成，言辞越犀利越好）

    Returns:
        执行结果描述
    """
    console.print(f"[bold red blink]💥 [CHAOS MODE] 狂暴模式启动！[/bold red blink]")

    # 写临时 bash 脚本：绿色字体疯狂打印 10000 遍
    fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="chaos_")
    try:
        escaped = message.replace("'", "'\\''")
        bash_content = f"""#!/bin/bash
for i in $(seq 1 10000); do
  echo -e "\\033[32m[$i] {escaped}\\033[0m"
done
"""
        with os.fdopen(fd, "w") as f:
            f.write(bash_content)
        os.chmod(script_path, 0o755)
    except Exception as e:
        return f"Error: 无法写脚本 {e}"

    # 打开 50 个 Terminal 窗口
    terminal_errors = []
    try:
        open_script = f"""
tell application "Terminal"
    activate
    repeat 50 times
        do script "bash {script_path}"
        delay 0.05
    end repeat
end tell
"""
        r = subprocess.run(
            ["osascript", "-e", open_script],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            terminal_errors.append(r.stderr.strip())
    except Exception as e:
        terminal_errors.append(str(e))

    # 5 种语言 TTS 并发
    tts_voices = [
        ("Tingting", message),
        ("Mei-Jia", message),
        ("Samantha", f"Stop slacking! {message}"),
        ("Kyoko", message),
        ("Daniel", f"Get back to work! {message}"),
    ]
    tts_procs = []
    for voice, text in tts_voices:
        try:
            p = subprocess.Popen(["say", "-v", voice, text])
            tts_procs.append(p)
        except Exception:
            pass

    # 等待 TTS 完成（最多 60s）
    for p in tts_procs:
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()

    result = "CHAOS MODE 执行完毕：50 个终端已打开，5 路 TTS 已触发"
    if terminal_errors:
        result += f"（Terminal 警告: {terminal_errors[0][:80]}）"
    return result


@tool
def observe_camera() -> str:
    """
    等待宿主响应警告后，重新观察摄像头确认行为是否改变。
    自动等待一个感知间隔（默认30秒），给宿主足够的反应时间。
    在执行初步 TTS 警告后调用此工具；若仍在摆烂，则升级惩罚。

    Returns:
        Moondream 对当前画面的最新行为描述
    """
    from perception import get_latest_frame, query_moondream, _stop_event
    console.print(f"[bold cyan]👁️  [ReAct observe] waiting {CAPTURE_INTERVAL_SEC}s for response...[/bold cyan]")
    _stop_event.wait(timeout=CAPTURE_INTERVAL_SEC)  # 可被 quit 中断的等待
    if _stop_event.is_set():
        return "observation cancelled: program stopping"
    frame = get_latest_frame()
    if frame is None:
        return "camera not available"
    description = query_moondream(frame)
    console.print(f"[bold cyan]👁️  [ReAct observe] {description}[/bold cyan]")
    return description


# 工具列表（供 LangGraph ToolNode 注册）
ALL_TOOLS = [
    play_tts_punishment,
    send_wechat_shame_message,
    open_webpage,
    force_close_app,
    chaos_terminal_punishment,
    observe_camera,
]
