"""
config.py — Global configuration for Cyber-Superego.

All tuneable parameters live here. Edit this file to customize behavior;
no changes needed in other source files.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a strict boolean environment flag with actionable errors."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: 1/0, true/false, yes/no, on/off"
    )


# ── Camera & Perception ───────────────────────────────────────
CAMERA_INDEX         = 0     # Webcam index (0 = default, 1/2 for external cameras)
CAPTURE_INTERVAL_SEC = 30    # Seconds between Moondream vision analyses
MEDIAPIPE_CONFIDENCE = 0.5   # Detection/tracking confidence threshold for MediaPipe

# ── OpenCV Display ────────────────────────────────────────────
GREEN_BOX_COLOR     = (0, 255, 0)   # Person bounding box color (BGR)
GREEN_BOX_THICKNESS = 2
TEXT_COLOR          = (0, 255, 0)
TEXT_FONT_SCALE     = 0.55
TEXT_THICKNESS      = 2
PERSON_BOX_PADDING  = 20            # Pixels added around detected person bbox

HAND_DOT_COLOR  = (0, 220, 255)     # Hand keypoint color (cyan-yellow, BGR)
HAND_LINE_COLOR = (0, 180, 255)     # Hand skeleton line color
HAND_DOT_RADIUS = 4
GESTURE_COLOR   = (0, 220, 255)     # Gesture label color

# ── Local Models (Ollama) ─────────────────────────────────────
OLLAMA_HOST            = "http://localhost:11434"
MOONDREAM_MODEL        = "moondream"                  # Vision model for behavior description
MOONDREAM_PROMPT       = "What is the person doing?"  # Prompt sent to Moondream each cycle
LOCAL_CLASSIFIER_MODEL = "qwen2.5:1.5b"               # Cerebellum: yes/no behavior classifier

# ── Cloud LLM (DeepSeek) ─────────────────────────────────────
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ── Side-effect safety ────────────────────────────────────────
# Potentially disruptive actions are disabled unless the operator explicitly opts in.
ENABLE_WECHAT_ACTIONS = _env_flag("WAKEUP_ENABLE_WECHAT_ACTIONS")
ENABLE_APP_TERMINATION = _env_flag("WAKEUP_ENABLE_APP_TERMINATION")
ENABLE_CHAOS_ACTIONS = _env_flag("WAKEUP_ENABLE_CHAOS_ACTIONS")

# ── Agent Memory & Persistence ────────────────────────────────
CHECKPOINT_DB_PATH   = "./superego.db"           # SQLite file for LangGraph checkpointer
DAILY_REPORT_PATH    = "./memory/daily_reports.md"
CONTEXT_MAX_MESSAGES = 20   # Max messages kept in LLM context window (trim_messages count)
SUMMARIZE_THRESHOLD  = 30   # Compress history into summary when message count exceeds this
REACT_MAX_ITERATIONS = 5    # Max DeepSeek ReAct loop rounds per punishment session

# ── WeChat Contacts ───────────────────────────────────────────
# Keys are internal aliases used by the LLM when calling send_wechat_shame_message.
# Values must exactly match the contact/group name as it appears in WeChat search.
# Fill in real contact names only when WAKEUP_ENABLE_WECHAT_ACTIONS is explicitly enabled.
WECHAT_CONTACTS = {
    "老妈":   "妈妈",
    "导师":   "导师",
    "班级群": "班级群",
}

# ── Console Log Prefixes (Rich markup) ───────────────────────
LOG_A = "[cyan][A][/cyan]"      # Perception node
LOG_B = "[yellow][B][/yellow]"  # Decision node
LOG_C = "[red][C][/red]"        # Execution node
LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
