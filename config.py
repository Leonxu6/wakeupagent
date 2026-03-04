"""
config.py — 全局配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── 摄像头 & 感知 ─────────────────────────────────────────────
CAMERA_INDEX = 0
CAPTURE_INTERVAL_SEC = 30
MEDIAPIPE_CONFIDENCE = 0.5

# ── 绘制 ──────────────────────────────────────────────────────
GREEN_BOX_COLOR    = (0, 255, 0)      # 人体框 (BGR)
GREEN_BOX_THICKNESS = 2
TEXT_COLOR         = (0, 255, 0)
TEXT_FONT_SCALE    = 0.55
TEXT_THICKNESS     = 2
PERSON_BOX_PADDING = 20

HAND_DOT_COLOR     = (0, 220, 255)    # 手部关键点颜色（青黄）
HAND_LINE_COLOR    = (0, 180, 255)    # 手部骨架连线
HAND_DOT_RADIUS    = 4
GESTURE_COLOR      = (0, 220, 255)    # 手势标签颜色

# ── Ollama / Moondream2 ───────────────────────────────────────
OLLAMA_HOST      = "http://localhost:11434"
MOONDREAM_MODEL  = "moondream"
MOONDREAM_PROMPT = "What is the person doing?"
LOCAL_CLASSIFIER_MODEL = "qwen2.5:1.5b"

# ── 云端 LLM (Node B, DeepSeek) ──────────────────────────────
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ── Agent 记忆 & 持久化 ────────────────────────────────────────
CHECKPOINT_DB_PATH  = "./superego.db"          # SQLite checkpointer
DAILY_REPORT_PATH   = "./memory/daily_reports.md"
CONTEXT_MAX_MESSAGES = 20                       # trim 阈值（条数）
SUMMARIZE_THRESHOLD  = 30                       # 触发摘要压缩的条数

# ── 日志前缀 ──────────────────────────────────────────────────
LOG_A = "[cyan][A][/cyan]"    # perception
LOG_B = "[yellow][B][/yellow]"  # decision
LOG_C = "[red][C][/red]"      # execution
# 向后兼容
LOG_PERCEPTION = LOG_A
LOG_DECISION   = LOG_B
LOG_EXECUTION  = LOG_C
