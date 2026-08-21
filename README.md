<div align="center">

# 🧠 WakeUpAgent

**A privacy-first, edge-cloud accountability agent for macOS.**

Local vision turns camera frames into short behavior descriptions; only text is sent to the cloud decision layer. Desktop side effects are centralized, testable, and disruptive actions are disabled by default.

[![CI](https://github.com/Leonxu6/wakeupagent/actions/workflows/ci.yml/badge.svg)](https://github.com/Leonxu6/wakeupagent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS-lightgrey?logo=apple)](https://apple.com/macos)

</div>

## What it does

WakeUpAgent watches a local camera feed, describes behavior with local models, classifies whether the observation needs attention, and feeds only text into a LangGraph decision loop. The agent can provide a spoken reminder, open a reference page, or re-observe the camera. Higher-impact desktop actions are **not model-visible by default** and require explicit operator opt-in.

The design priorities are:

- **Privacy:** raw camera frames remain local.
- **Fail-closed automation:** contact and app-termination actions default to disabled.
- **Testable boundaries:** browser, subprocess, camera, and UI automation are isolated in `tools.py`.
- **Durable context:** LangGraph checkpoints keep state across restarts.
- **Low cloud usage:** local perception and classification reduce unnecessary remote calls.

## Architecture

```text
Webcam
  │
  ├─ MediaPipe pose + gesture overlay
  │
  └─ Moondream local VLM ──> short text observation
                              │
                              ├─ qwen2.5:1.5b local classifier
                              │
                              └─ DeepSeek + LangGraph decision loop
                                           │
                                           └─ validated local tools
                                              ├─ TTS reminder
                                              ├─ browser page
                                              ├─ re-observe camera
                                              └─ explicit opt-in actions
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component boundaries and [docs/SAFETY.md](docs/SAFETY.md) for side-effect rules.

## Requirements

- macOS for the desktop automation integrations
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- a DeepSeek API key for the cloud decision layer

## Setup

```bash
git clone https://github.com/Leonxu6/wakeupagent.git
cd wakeupagent
uv sync --frozen
cp .env.example .env
```

Pull the local models:

```bash
ollama pull moondream
ollama pull qwen2.5:1.5b
```

Set `DEEPSEEK_API_KEY` in `.env`, then run the offline diagnostic before starting the camera loop:

```bash
uv run main.py --doctor
```

The repository includes the MediaPipe `pose_landmarker_lite.task` and `gesture_recognizer.task` assets. The doctor command checks that both are present without contacting external services.

## Run modes

```bash
# Live perception + graph workflow
uv run main.py

# One mock graph cycle without camera capture
uv run main.py --graph

# Offline setup diagnostics
uv run main.py --doctor

# Unit tests
uv run --with pytest pytest -q
```

## Safe defaults

Potentially disruptive integrations are disabled unless explicitly enabled before process startup:

```dotenv
WAKEUP_ENABLE_WECHAT_ACTIONS=false
WAKEUP_ENABLE_APP_TERMINATION=false
WAKEUP_ENABLE_CHAOS_ACTIONS=false
```

`WAKEUP_ENABLE_CHAOS_ACTIONS` remains documented for migration visibility, but terminal-flooding automation has been removed from the model tool registry and the compatibility function is a safe no-op. Contact and app-termination actions appear in the tool registry only after their dedicated flags are enabled.

Accepted boolean values are `1/0`, `true/false`, `yes/no`, and `on/off`. Invalid values fail during configuration loading instead of silently enabling an action.

Read [docs/CONFIGURATION.md](docs/CONFIGURATION.md) before changing these settings.

## Privacy model

Camera frames are consumed by local perception code. The cloud decision layer receives text observations, graph memory, and tool results rather than raw images. Runtime state is stored locally in `superego.db` and `memory/daily_reports.md`, resolved relative to the project root so launching from another working directory does not scatter private state.

Do not commit `.env`, generated reports, checkpoints, contact names, screenshots containing personal data, or camera frames. See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Repository map

```text
wakeupagent/
├── main.py                  # CLI and runtime wiring
├── graph.py                 # LangGraph state machine and durable memory
├── perception.py            # Camera, MediaPipe, local VLM/classifier
├── tools.py                 # Validated side-effect boundary
├── config.py                # Runtime configuration and opt-in flags
├── doctor.py                # Offline setup diagnostics
├── runtime_paths.py         # Stable project-relative runtime paths
├── tests/                   # Unit/regression coverage
├── docs/                    # Architecture, safety, and configuration
└── .github/workflows/ci.yml # Push/PR verification
```

## Development

Before a pull request:

```bash
uv sync --frozen
uv run --with pytest pytest -q
```

Normal CI compiles the Python entry points and runs the test suite without touching a real camera, browser, WeChat client, or desktop process. External side effects belong behind mocks in unit tests.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations and [SECURITY.md](SECURITY.md) for security-sensitive changes.

## License

MIT. See [LICENSE](LICENSE).
