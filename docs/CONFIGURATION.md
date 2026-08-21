# Configuration

WakeUpAgent keeps ordinary tuning constants in `config.py` and secrets or safety-sensitive switches in environment variables.

## Required secret

`DEEPSEEK_API_KEY` supplies the cloud decision model credential. Keep it in `.env` or the process environment and never commit the real value.

## Disruptive action flags

All three flags default to `false`:

| Variable | Effect when explicitly enabled |
| --- | --- |
| `WAKEUP_ENABLE_WECHAT_ACTIONS` | Allows configured contact automation |
| `WAKEUP_ENABLE_APP_TERMINATION` | Allows local app termination tools |
| `WAKEUP_ENABLE_CHAOS_ACTIONS` | Allows legacy chaos automation |

Accepted boolean values are `1/0`, `true/false`, `yes/no`, and `on/off`, case-insensitively. Any other value fails during configuration import rather than silently enabling an action.

## Local model settings

`OLLAMA_HOST`, `MOONDREAM_MODEL`, `MOONDREAM_PROMPT`, and `LOCAL_CLASSIFIER_MODEL` are ordinary Python constants. Keep model names and endpoints local unless a deployment needs a different host.

## Persistence

`CHECKPOINT_DB_PATH` points to the LangGraph SQLite checkpoint. `DAILY_REPORT_PATH` stores generated daily summaries. Treat both as private local state; they are runtime data, not source files.
