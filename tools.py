"""Bounded local tools used by the WakeUpAgent LangGraph execution node."""
from __future__ import annotations

import os
import subprocess
import webbrowser

from langchain_core.tools import tool
from rich.console import Console
from rich.markup import escape

from config import CAPTURE_INTERVAL_SEC
from safety import require_app_name, require_http_url, require_text
from settings import env_bool

console = Console()
_TTS_VOICE = "Tingting"


def _feature_enabled(name: str) -> bool:
    try:
        return env_bool(name, False)
    except ValueError:
        return False


def _error(exc: ValueError) -> str:
    return f"Error: {exc}"


def _bounded_detail(value: object, *, limit: int = 500) -> str:
    """Keep driver/subprocess failures compact and single-line for agent context."""
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001
        rendered = value.__class__.__name__
    text = " ".join(rendered.split())
    return text[:limit] if text else "unknown error"


def _observation_text(value: object, *, limit: int = 1000) -> str:
    """Normalize a local vision response before logging or returning it to the graph."""
    if not isinstance(value, str):
        raise ValueError("camera description must be text")
    text = " ".join(value.split())
    if not text:
        raise ValueError("camera description must not be empty")
    return text[:limit]


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@tool
def play_tts_punishment(text: str) -> str:
    """Speak one short local reminder when local TTS has been explicitly enabled."""
    if not _feature_enabled("WAKEUP_ALLOW_TTS"):
        return "Error: local TTS is disabled; set WAKEUP_ALLOW_TTS=true to opt in"
    try:
        text = require_text(text, field="text", max_length=200)
    except ValueError as exc:
        return _error(exc)
    console.print(f"[bold yellow]🔊 [TTS] {escape(text)}[/bold yellow]")
    try:
        subprocess.run(["say", "-v", _TTS_VOICE, text], timeout=60, check=True)
        return "TTS 播放完毕"
    except FileNotFoundError:
        return "Error: say 命令不存在（仅支持 macOS）"
    except subprocess.TimeoutExpired:
        return "Error: TTS 播放超时"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["say", text], timeout=60, check=True)
            return "TTS 播放完毕（默认声音）"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {_bounded_detail(exc)}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {_bounded_detail(exc)}"


@tool
def send_wechat_shame_message(target: str, message: str) -> str:
    """Send a user-approved WeChat message when external messaging is enabled."""
    if not _feature_enabled("WAKEUP_ALLOW_EXTERNAL_MESSAGING"):
        return "Error: external messaging is disabled; set WAKEUP_ALLOW_EXTERNAL_MESSAGING=true to opt in"
    try:
        target = require_text(target, field="target", max_length=40)
        message = require_text(message, field="message", max_length=500)
    except ValueError as exc:
        return _error(exc)

    from config import WECHAT_CONTACTS
    contact = WECHAT_CONTACTS.get(target)
    if not contact:
        return f"Error: 不支持的 target '{target}'，只能用: {list(WECHAT_CONTACTS.keys())}"
    try:
        contact = require_text(contact, field="contact", max_length=100)
    except ValueError as exc:
        return _error(exc)

    # Log only the model-facing alias. The resolved address-book name is private
    # configuration and should not be copied into terminal scrollback.
    console.print(f"[bold yellow]🦾 [WeChat] → alias {escape(target)}[/bold yellow]")
    script = f'''
do shell script "open -a WeChat"
delay 2.0
tell application "System Events"
    tell process "WeChat"
        keystroke "f" using {{command down}}
        delay 1.0
        keystroke "a" using {{command down}}
        set the clipboard to "{_escape_applescript(contact)}"
        keystroke "v" using {{command down}}
        delay 2.0
        keystroke return
        delay 1.0
        set the clipboard to "{_escape_applescript(message)}"
        keystroke "v" using {{command down}}
        delay 0.5
        keystroke return
    end tell
end tell
'''
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"Error: {_bounded_detail(result.stderr or 'osascript 执行失败')}"
        return f"已向别名 {target} 发送消息"
    except subprocess.TimeoutExpired:
        return "Error: 微信操作超时"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {_bounded_detail(exc)}"


@tool
def open_webpage(url: str) -> str:
    """Open a validated HTTP(S) URL after browser control has been explicitly enabled."""
    if not _feature_enabled("WAKEUP_ALLOW_BROWSER_CONTROL"):
        return "Error: browser control is disabled; set WAKEUP_ALLOW_BROWSER_CONTROL=true to opt in"
    try:
        url = require_http_url(url)
    except ValueError as exc:
        return _error(exc)
    console.print(f"[bold cyan]🌐 [browser] opening {escape(url)}[/bold cyan]")
    try:
        if not webbrowser.open(url):
            return f"Error: 浏览器未能打开：{url}"
        return f"已在浏览器中打开：{url}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {_bounded_detail(exc)}"


@tool
def force_close_app(app_name: str) -> str:
    """Request a graceful quit for one explicitly named app when process control is enabled."""
    if not _feature_enabled("WAKEUP_ALLOW_PROCESS_CONTROL"):
        return "Error: process control is disabled; set WAKEUP_ALLOW_PROCESS_CONTROL=true to opt in"
    try:
        app_name = require_app_name(app_name)
    except ValueError as exc:
        return _error(exc)
    console.print(f"[bold yellow]⏹ [process] requesting app quit: {escape(app_name)}[/bold yellow]")
    escaped_name = _escape_applescript(app_name)
    try:
        result = subprocess.run(
            ["osascript", "-e", f'tell application "{escaped_name}" to quit'],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: 无法请求 {app_name} 退出：{_bounded_detail(exc)}"
    if result.returncode == 0:
        return f"已请求 {app_name} 正常退出"
    return f"Error: 无法请求 {app_name} 退出：{_bounded_detail(result.stderr or 'application did not quit')}"


@tool
def chaos_terminal_punishment(message: str) -> str:
    """Legacy compatibility entry point for the removed destructive chaos mode."""
    return "Error: chaos mode is disabled because it creates disruptive terminal/process side effects"


@tool
def observe_camera() -> str:
    """Wait one perception interval and return the latest local vision summary."""
    from perception import _stop_event, get_latest_frame, query_moondream
    console.print(f"[bold cyan]👁️  [observe] waiting {CAPTURE_INTERVAL_SEC}s for response...[/bold cyan]")
    _stop_event.wait(timeout=CAPTURE_INTERVAL_SEC)
    if _stop_event.is_set():
        return "observation cancelled: program stopping"
    try:
        frame = get_latest_frame()
    except Exception as exc:  # noqa: BLE001
        return f"Error: camera frame unavailable: {_bounded_detail(exc)}"
    if frame is None:
        return "camera not available"
    try:
        description = _observation_text(query_moondream(frame))
    except ValueError as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001
        return f"Error: camera description failed: {_bounded_detail(exc)}"
    console.print(f"[bold cyan]👁️  [observe] {escape(description)}[/bold cyan]")
    return description


ALL_TOOLS = [play_tts_punishment, send_wechat_shame_message, open_webpage, force_close_app, observe_camera]
