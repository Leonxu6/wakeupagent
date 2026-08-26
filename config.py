"""
config.py — Global configuration for Cyber-Superego.

Defaults are suitable for local development. Runtime-specific values can be
overridden through validated environment variables without editing source.
"""
from dotenv import load_dotenv

from settings import env_float, env_http_url, env_int, env_json_string_map, env_path, env_secret, env_text

load_dotenv()

# ── Camera & Perception ───────────────────────────────────────
CAMERA_INDEX = env_int("WAKEUP_CAMERA_INDEX", 0, minimum=0, maximum=32)
CAPTURE_INTERVAL_SEC = env_float("WAKEUP_CAPTURE_INTERVAL_SEC", 30.0, minimum=0.1, maximum=3600)
MEDIAPIPE_CONFIDENCE = env_float("WAKEUP_MEDIAPIPE_CONFIDENCE", 0.5, minimum=0.0, maximum=1.0)

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
OLLAMA_HOST = env_http_url("OLLAMA_HOST", "http://localhost:11434")
MOONDREAM_MODEL = env_text("MOONDREAM_MODEL", "moondream", max_length=120)
MOONDREAM_PROMPT = env_text("MOONDREAM_PROMPT", "What is the person doing?", max_length=500)
LOCAL_CLASSIFIER_MODEL = env_text("LOCAL_CLASSIFIER_MODEL", "qwen2.5:1.5b", max_length=120)

# ── Cloud LLM (DeepSeek) ─────────────────────────────────────
DEEPSEEK_API_KEY = env_secret("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = env_text("DEEPSEEK_MODEL", "deepseek-chat", max_length=120)
DEEPSEEK_BASE_URL = env_http_url("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── Agent Memory & Persistence ────────────────────────────────
CHECKPOINT_DB_PATH = env_path("WAKEUP_CHECKPOINT_DB_PATH", "./superego.db")
DAILY_REPORT_PATH = env_path("WAKEUP_DAILY_REPORT_PATH", "./memory/daily_reports.md")
CONTEXT_MAX_MESSAGES = env_int("WAKEUP_CONTEXT_MAX_MESSAGES", 20, minimum=1, maximum=500)
SUMMARIZE_THRESHOLD = env_int("WAKEUP_SUMMARIZE_THRESHOLD", 30, minimum=2, maximum=2000)
REACT_MAX_ITERATIONS = env_int("WAKEUP_REACT_MAX_ITERATIONS", 5, minimum=1, maximum=20)

# ── WeChat Contacts ───────────────────────────────────────────
# Aliases are used by the model; values must exactly match WeChat search names.
# Keep the repository default empty so private contact names are always supplied
# explicitly by the local operator, for example:
# WAKEUP_WECHAT_CONTACTS_JSON='{"family":"Mom","mentor":"Dr Xu"}'
WECHAT_CONTACTS = env_json_string_map(
    "WAKEUP_WECHAT_CONTACTS_JSON",
    {},
    max_entries=50,
)

# ── Console Log Prefixes (Rich markup) ────────────────────────
LOG_A = "[cyan][A][/cyan]"      # Perception node
LOG_B = "[yellow][B][/yellow]"  # Decision node
LOG_C = "[red][C][/red]"        # Execution node
LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
