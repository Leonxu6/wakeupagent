"""Bounded local tools used by the WakeUpAgent LangGraph execution node."""
from __future__ import annotations

import os
import subprocess
import unicodedata
import webbrowser
from urllib.parse import urlsplit, urlunsplit

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
    """Keep driver/subprocess failures compact and single-line for local logs."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001
        rendered = value.__class__.__name__
    rendered = "".join(ch if ord(ch) >= 32 and ord(ch) != 127 else " " for ch in rendered)
    text = " ".join(rendered.split())
    return text[:limit] if text else "unknown error"


def _safe_url_label(url: str) -> str:
    """Return a browser URL label without path/query/fragment secrets."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _observation_text(value: object, *, limit: int = 1000) -> str:
    """Normalize a local vision response before logging or returning it to the graph."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("observation limit must be a positive integer")
    if not isinstance(value, str):
        raise ValueError("camera description must be text")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("camera description contains control characters")
    text = " ".join(value.split())
    if not text:
        raise ValueError("camera description must not be empty")
    return text[:limit]


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _alias_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _resolve_contact_alias(target: str, contacts: object) -> str | None:
    """Resolve one normalized configured alias without exposing contact values."""
    if not isinstance(contacts, dict):
        return None
    folded = _alias_identity(target)
    matches = [
        value
        for alias, value in contacts.items()
        if isinstance(alias, str) and _alias_identity(alias) == folded
    ]
    if len(matches) != 1:
        return None
    return matches[0]


@tool
def play_tts_punishment(text: str) -> str:
    """Speak one short local reminder when local TTS has been explicitly enabled."""
    if not _feature_enabled("WAKEUP_ALLOW_TTS"):
        return "Error: local TTS is disabled; set WAKEUP_ALLOW_TTS=true to opt in"
    try:
        text = require_text(text, field="text", max_length=200)
    except ValueError as exc:
        return _error(exc)
    console.print(f"[bold yellow]🔊 [TTS] speaking {len(text)} characters[/bold yellow]")
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
        except Exception:  # noqa: BLE001
            return "Error: TTS 播放失败"
    except Exception:  # noqa: BLE001
        return "Error: TTS 播放失败"


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
    contact = _resolve_contact_alias(target, WECHAT_CONTACTS)
    if not contact:
        return "Error: unsupported messaging target"
    try:
        contact = require_text(contact, field="contact", max_length=100)
    except ValueError as exc:
        return _error(exc)

    console.print("[bold yellow]🦾 [WeChat] sending approved message[/bold yellow]")
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
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return "Error: 微信自动化执行失败"
        return "消息发送完成"
    except FileNotFoundError:
        return "Error: osascript 命令不存在（仅支持 macOS）"
    except subprocess.TimeoutExpired:
        return "Error: 微信操作超时"
    except Exception:  # noqa: BLE001
        return "Error: 微信自动化执行失败"


@tool
def open_webpage(url: str) -> str:
    """Open a validated HTTP(S) URL after browser control has been explicitly enabled."""
    if not _feature_enabled("WAKEUP_ALLOW_BROWSER_CONTROL"):
        return "Error: browser control is disabled; set WAKEUP_ALLOW_BROWSER_CONTROL=true to opt in"
    try:
        url = require_http_url(url)
    except ValueError as exc:
        return _error(exc)
    label = _safe_url_label(url)
    console.print(f"[bold cyan]🌐 [browser] opening {escape(label)}[/bold cyan]")
    try:
        if not webbrowser.open(url):
            return f"Error: 浏览器未能打开：{label}"
        return f"已在浏览器中打开：{label}"
    except Exception:  # noqa: BLE001
        return f"Error: 浏览器打开失败：{label}"


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
            check=False,
        )
    except FileNotFoundError:
        return "Error: osascript 命令不存在（仅支持 macOS）"
    except subprocess.TimeoutExpired:
        return f"Error: 请求 {app_name} 退出超时"
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red][process] quit request failed: {escape(_bounded_detail(exc))}[/red]")
        return f"Error: 无法请求 {app_name} 退出"
    if result.returncode == 0:
        return f"已请求 {app_name} 正常退出"
    console.print(f"[red][process] quit request failed: {escape(_bounded_detail(result.stderr or 'application did not quit'))}[/red]")
    return f"Error: 无法请求 {app_name} 退出"


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
        console.print(f"[red][observe] camera frame unavailable: {escape(_bounded_detail(exc))}[/red]")
        return "Error: camera frame unavailable"
    if frame is None:
        return "camera not available"
    try:
        description = _observation_text(query_moondream(frame))
    except ValueError as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red][observe] camera description failed: {escape(_bounded_detail(exc))}[/red]")
        return "Error: camera description failed"
    console.print(f"[bold cyan]👁️  [observe] description captured ({len(description)} chars)[/bold cyan]")
    return description


ALL_TOOLS = [play_tts_punishment, send_wechat_shame_message, open_webpage, force_close_app, observe_camera]
