# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run perception loop (live camera, full system)
uv run main.py

# Test LangGraph flow with mock state (no camera needed)
uv run main.py --graph

# Run perception node directly (camera only, no graph)
uv run python perception.py

# Install dependencies
uv sync

# Add a package
uv add <package>
```

## Environment

Requires a `.env` file with:
```
DEEPSEEK_API_KEY=...
```

External services that must be running:
- **Ollama** (`open /Applications/Ollama.app`) serving `moondream` and `qwen2.5:1.5b`
- Local model files in project root: `pose_landmarker_lite.task`, `gesture_recognizer.task`

System has a SOCKS proxy (`all_proxy=socks5://127.0.0.1:7897`), which is why `httpx[socks]` is a dependency.

## Architecture

**Cyber-Superego** is an edge-cloud hybrid supervisor that monitors user behavior via webcam and intervenes with escalating punishments when it detects procrastination.

### LangGraph Flow

```
START → [R] daily_reset → [A] perception → (route) → [B] decision ⇌ [C] execution (ReAct loop) → END
```

- **[R] `daily_reset_node`** (`graph.py`): Detects day change, generates daily report via LLM, clears message history, resets counters.
- **[A] `perception_node`** (`graph.py`): Wraps vision data from `perception.py` into a `HumanMessage` for the LLM. Routes to `[B]` only if `should_escalate=True`.
- **[B] `decision_node`** (`graph.py`): DeepSeek ReAct loop. Calls tools for graduated punishment, handles voice appeals mid-loop. Max `REACT_MAX_ITERATIONS` rounds.
- **[C] `ToolNode`** (`graph.py` + `tools.py`): Executes tools; loops back to `[B]`. Currently all tools are **MOCK**.

State is persisted via `SqliteSaver` to `superego.db` using a fixed `thread_id="superego_main"`.

### Perception Pipeline (`perception.py`)

Real-time camera loop:
1. Every frame: MediaPipe `PoseLandmarker` (pose) + `GestureRecognizer` (hands) for display
2. Every `CAPTURE_INTERVAL_SEC` (30s): snapshot → `query_moondream()` → `classify_behavior()` (qwen2.5:1.5b cerebellum)
3. `classify_behavior()` returns `(is_healthy, should_escalate)`. Logic: Python keyword pre-filter → time exemption (0–7am) → qwen yes/no classification
4. `_latest_raw_frame` is a module-level global used by `observe_camera` tool for ReAct re-observation

**Critical**: save `raw_frame = frame.copy()` before drawing UI overlays — Moondream sees the clean frame.

### Voice Module (`voice.py`)

Single audio thread (prevents sounddevice device conflicts):
- Continuously records `VOICE_CHUNK_SEC` windows, checks RMS, transcribes with faster-whisper `tiny`
- On wake word "乌鲁鲁": records full speech with VAD silence detection → STT with `small` model
- **Two paths after wake**:
  - If `_graph_active=True` → queue as appeal (`_appeal_text`), injected as `HumanMessage` in next `decision_node` iteration
  - If graph idle → classify intent (`goal`/`exempt`/`observe`) → fire `on_voice_input` callback in separate thread
- Every `AMBIENT_INTERVAL_SEC` (30s): sample ambient audio, transcribe, store in `ambient_text`

### Key Data Flow

`main.py` wires everything:
- `on_vision` callback: timer-triggered perception results → `graph.stream()`
- `on_voice_input` callback: voice-triggered → `graph.stream()` with `trigger_source="voice"`
- `get_context()`: returns rolling window of recent observations + LLM verdicts → fed to qwen cerebellum
- `set_graph_active(True/False)` bracketing `graph.stream()` enables the appeal mechanism

### AgentState Fields

Key non-obvious fields:
- `should_escalate`: set by qwen cerebellum; gates whether `[A]→[B]` route fires
- `react_iterations`: ReAct loop counter, reset to 0 at END; repaired on crash restart
- `consecutive_healthy`: increments only on `trigger_source="timer"` AND `iter==0` (no tools called)
- `session_goal` / `voice_rules`: set by voice intent, persisted via checkpointer, cleared on day reset
- `conversation_summary`: rolling compressed memory; daily report becomes next day's initial summary

### Tools (`tools.py`)

All tools are currently MOCK. The `observe_camera` tool uses `_stop_event.wait(timeout=30)` so it can be interrupted on quit. When implementing real tools, each must `try/except` and return an error string — `ToolNode` does not catch exceptions.

`parallel_tool_calls=False` is set on the LLM binding — `observe_camera` must never run in parallel with other tools.

## Configuration (`config.py`)

All tunable constants are in `config.py`. Notable:
- `REACT_MAX_ITERATIONS = 5` — max ReAct loops before forced END
- `SUMMARIZE_THRESHOLD = 30` — message count that triggers memory compression
- `CONTEXT_MAX_MESSAGES = 20` — `trim_messages` keeps this many recent messages
- `LOCAL_CLASSIFIER_MODEL = "qwen2.5:1.5b"` — cerebellum model (must be pulled in Ollama)
