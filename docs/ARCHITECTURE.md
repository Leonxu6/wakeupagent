# Architecture

WakeUpAgent is split into four runtime concerns so that perception, reasoning, persistence, and side effects can be tested independently.

## Perception

`perception.py` owns camera/model interaction and emits short observations. It should not perform irreversible actions. The perception loop hands observations to `main.py`, which builds graph inputs without overwriting checkpointed daily counters.

## Decision graph

`graph.py` owns the LangGraph state machine. `daily_reset` handles day rollover, `perception` records the observation, `decision` asks the configured model what to do, and `execution` runs explicitly registered tools. The SQLite checkpointer keeps durable counters and conversation state across process restarts.

## Tools

`tools.py` is the side-effect boundary. Browser, TTS, camera observation, contact automation, and app control live here so tests can patch external APIs. Disruptive tools must remain opt-in and must validate arguments before invoking operating-system commands.

## Configuration

`config.py` contains operator-tunable settings and strict environment flags. Secrets come from environment variables. Generated reports and the SQLite checkpoint are local runtime state and should not be committed.

## Testing boundaries

Unit tests should prefer pure helpers and mocks. Camera hardware, browsers, WeChat, TTS, and macOS UI automation are integration boundaries and must not be exercised by normal CI.
