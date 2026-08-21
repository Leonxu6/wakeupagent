"""
tools.py — Node C local execution tools.

Side effects are deliberately centralized here so they can be validated, mocked,
and disabled independently from perception and reasoning.
"""
import re
import subprocess
import webbrowser
from urllib.parse import urlparse

from langchain_core.tools import tool
from rich.console import Console
from config import (
    CAPTURE_INTERVAL_SEC,
    ENABLE_APP_TERMINATION,
    ENABLE_WECHAT_ACTIONS,
)

console = Console()

_TTS_VOICE = "Tingting"
_APP_NAME_RE = re.compile(r"[\w .+()-]{1,80}\Z", re.UNICODE)


def _validated_text(value, *, name: str, max_length: int) -> str:
    """Validate text before it crosses into a desktop automation boundary."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _has_http_url_target(url: str) -> bool:
    """Return whether a URL is a clean HTTP(S) target with a hostname."""
    if not isinstance(url, str) or not url or url != url.strip():
        return False
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return False
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )
    except (AttributeError, ValueError):
        return False


@tool
def play_tts_punishment(text: str) -> str:
    """Read a short accountability reminder through macOS text-to-speech."""
    try:
        text = _validated_text(text, name="text", max_length=300)
    except ValueError as exc:
        return f"Error: {exc}"

    console.print(f"[bold yellow]🔊 [TTS] {text}[/bold yellow]")
    try:
        subprocess.run(["say", "-v", _TTS_VOICE, text], timeout=60, check=True)
        return f"TTS 播放完毕：{text}"
    except FileNotFoundError:
        return "Error: say 命令不存在（仅支持 macOS）"
    except subprocess.TimeoutExpired:
        return "Error: TTS 播放超时"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["say", text], timeout=60, check=True)
            return f"TTS 播放完毕（默认声音）：{text}"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@tool
def send_wechat_shame_message(target: str, message: str) -> str:
    """Send an operator-configured accountability message through WeChat.

    This tool is unavailable unless `WAKEUP_ENABLE_WECHAT_ACTIONS=true` was set
    before the process started. Targets are aliases defined in `WECHAT_CONTACTS`.
    """
    if not ENABLE_WECHAT_ACTIONS:
        return "Error: WeChat automation is disabled; set WAKEUP_ENABLE_WECHAT_ACTIONS=true to opt in"

    try:
        target = _validated_text(target, name="target", max_length=32)
        message = _validated_text(message, name="message", max_length=500)
    except ValueError as exc:
        return f"Error: {exc}"

    from config import WECHAT_CONTACTS

    contact = WECHAT_CONTACTS.get(target)
    if not contact:
        return f"Error: unsupported target '{target}'; allowed aliases: {list(WECHAT_CONTACTS.keys())}"
    try:
        contact = _validated_text(contact, name="contact", max_length=80)
    except ValueError as exc:
        return f"Error: invalid configured contact: {exc}"

    console.print(f"[bold yellow]🦾 [WeChat] → {target}({contact})[/bold yellow]")

    def _esc(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

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
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip() or 'osascript 执行失败'}"
        return f"已向 {target}({contact}) 发送提醒"
    except subprocess.TimeoutExpired:
        return "Error: 微信操作超时"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@tool
def open_webpage(url: str) -> str:
    """Open a validated HTTP(S) learning or reference page in the default browser."""
    if not _has_http_url_target(url):
        return "Error: URL 必须是无凭据、包含主机名的 http:// 或 https:// 地址"

    console.print(f"[bold cyan]🌐 [browser] open {url}[/bold cyan]")
    try:
        if not webbrowser.open(url):
            return f"Error: 浏览器未能打开：{url}"
        return f"已在浏览器中打开：{url}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@tool
def force_close_app(app_name: str) -> str:
    """Close one exact local application name when the operator explicitly opts in."""
    if not ENABLE_APP_TERMINATION:
        return "Error: app termination is disabled; set WAKEUP_ENABLE_APP_TERMINATION=true to opt in"
    try:
        app_name = _validated_text(app_name, name="app_name", max_length=80)
    except ValueError as exc:
        return f"Error: {exc}"
    if not _APP_NAME_RE.fullmatch(app_name):
        return "Error: app_name contains unsupported characters"

    console.print(f"[bold yellow]⏹ [app] close {app_name}[/bold yellow]")
    escaped = app_name.replace("\\", "\\\\").replace('"', '\\"')
    errors = []

    try:
        result = subprocess.run(
            ["osascript", "-e", f'tell application "{escaped}" to quit'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"已通过 osascript 关闭 {app_name}"
        errors.append(f"osascript: {result.stderr.strip()}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"osascript: {exc}")

    try:
        result = subprocess.run(
            ["killall", app_name], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return f"已通过 killall 关闭 {app_name}"
        errors.append(f"killall: {result.stderr.strip()}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"killall: {exc}")

    return f"Error: 无法关闭 {app_name}。详情: {'; '.join(errors)}"


@tool
def chaos_terminal_punishment(message: str) -> str:
    """Deprecated compatibility stub for the removed terminal-flooding action."""
    return "Error: chaos terminal automation has been removed for safety and reliability"


@tool
def observe_camera() -> str:
    """Wait one perception interval, then return the latest local camera description."""
    from perception import get_latest_frame, query_moondream, _stop_event

    console.print(
        f"[bold cyan]👁️  [observe] waiting {CAPTURE_INTERVAL_SEC}s for response...[/bold cyan]"
    )
    _stop_event.wait(timeout=CAPTURE_INTERVAL_SEC)
    if _stop_event.is_set():
        return "observation cancelled: program stopping"
    frame = get_latest_frame()
    if frame is None:
        return "camera not available"
    description = query_moondream(frame)
    console.print(f"[bold cyan]👁️  [observe] {description}[/bold cyan]")
    return description


# Only low-impact tools are model-visible by default. Higher-impact actions appear
# only when the operator opted in before process startup. The deprecated chaos tool
# is intentionally never registered.
ALL_TOOLS = [play_tts_punishment, open_webpage, observe_camera]
if ENABLE_WECHAT_ACTIONS:
    ALL_TOOLS.append(send_wechat_shame_message)
if ENABLE_APP_TERMINATION:
    ALL_TOOLS.append(force_close_app)
