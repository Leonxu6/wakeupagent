"""
config.py — Global configuration for Cyber-Superego.

All tuneable parameters live here. Edit this file to customize behavior;
no changes needed in other source files.
"""
import os
from dotenv import load_dotenv

load_dotenv()

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

# ── Agent Memory & Persistence ────────────────────────────────
CHECKPOINT_DB_PATH   = "./superego.db"           # SQLite file for LangGraph checkpointer
DAILY_REPORT_PATH    = "./memory/daily_reports.md"
CONTEXT_MAX_MESSAGES = 20   # Max messages kept in LLM context window (trim_messages count)
SUMMARIZE_THRESHOLD  = 30   # Compress history into summary when message count exceeds this
REACT_MAX_ITERATIONS = 5    # Max DeepSeek ReAct loop rounds per punishment session

# ── WeChat Contacts ───────────────────────────────────────────
# Keys are internal aliases used by the LLM when calling send_wechat_shame_message.
# Values must exactly match the contact/group name as it appears in WeChat search.
# ⚠️  Fill in your real contact names before running — defaults are placeholders.
WECHAT_CONTACTS = {
    "老妈":   "妈妈",       # e.g. "Mom" — your mother's WeChat display name
    "导师":   "导师",       # e.g. "Prof. Zhang" — your supervisor's WeChat name
    "班级群": "班级群",     # e.g. "Class 2024" — your class group chat name
}

# ── Console Log Prefixes (Rich markup) ───────────────────────
LOG_A = "[cyan][A][/cyan]"      # Perception node
LOG_B = "[yellow][B][/yellow]"  # Decision node
LOG_C = "[red][C][/red]"        # Execution node
LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
