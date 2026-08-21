# Runtime configuration

WakeUpAgent loads `.env` before evaluating `config.py`. Runtime overrides are parsed by `settings.py` so invalid values fail early instead of reaching camera, model, or persistence libraries.

## Camera and perception

| Variable | Default | Accepted range |
| --- | --- | --- |
| `WAKEUP_CAMERA_INDEX` | `0` | integer `0..32` |
| `WAKEUP_CAPTURE_INTERVAL_SEC` | `30` | finite number `0.1..3600` |
| `WAKEUP_MEDIAPIPE_CONFIDENCE` | `0.5` | finite number `0..1` |

## Models

`OLLAMA_HOST` and `DEEPSEEK_BASE_URL` must be HTTP(S) URLs with a hostname and no embedded credentials. Model names reject empty, padded, overlong, and control-character values.

## Persistence

`WAKEUP_CHECKPOINT_DB_PATH` and `WAKEUP_DAILY_REPORT_PATH` support `~` expansion. Context and iteration limits are bounded integers so a typo cannot silently allocate unbounded work.

## Side effects

External messaging and process control are disabled unless explicitly enabled with:

```text
WAKEUP_ALLOW_EXTERNAL_MESSAGING=true
WAKEUP_ALLOW_PROCESS_CONTROL=true
```

Keep both disabled while evaluating the project or running tests.

## Installation check

Run:

```bash
uv run main.py --check
```

The check verifies model files, persistence directories, endpoint shapes, and whether the optional DeepSeek key is configured. It does not open the camera or contact network services.
