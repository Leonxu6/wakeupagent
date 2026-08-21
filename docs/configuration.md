# Runtime configuration

WakeUpAgent loads `.env` before evaluating `config.py`. Runtime overrides are parsed by `settings.py` so invalid values fail early instead of reaching camera, model, persistence, or HTTP client libraries.

## Camera and perception

| Variable | Default | Accepted range |
| --- | --- | --- |
| `WAKEUP_CAMERA_INDEX` | `0` | integer `0..32` |
| `WAKEUP_CAPTURE_INTERVAL_SEC` | `30` | finite number `0.1..3600` |
| `WAKEUP_MEDIAPIPE_CONFIDENCE` | `0.5` | finite number `0..1` |

## Models and cloud credentials

`OLLAMA_HOST` and `DEEPSEEK_BASE_URL` must be clean HTTP(S) service base URLs with a hostname, no embedded credentials, and no query string or fragment. Model names reject empty, padded, overlong, and control-character values.

`DEEPSEEK_API_KEY` is optional for local-only diagnostics and may be left empty. When set, it rejects leading/trailing whitespace, control characters, and implausibly long values before the key can reach an HTTP authorization header. The checked-in `.env.example` intentionally leaves the key empty so copying it does not look like a configured credential.

## Persistence

`WAKEUP_CHECKPOINT_DB_PATH` and `WAKEUP_DAILY_REPORT_PATH` support `~` expansion. Context and iteration limits are bounded integers so a typo cannot silently allocate unbounded work.

## Side effects

External messaging and process control are disabled unless explicitly enabled with:

```text
WAKEUP_ALLOW_EXTERNAL_MESSAGING=true
WAKEUP_ALLOW_PROCESS_CONTROL=true
```

Keep both disabled while evaluating the project or running tests. The diagnostics command also parses these flags and reports malformed boolean values instead of silently treating a typo as a valid configuration.

## Installation check

Run:

```bash
uv run main.py --check
```

The check verifies model files, persistence directories, endpoint shapes, optional cloud credentials, and side-effect feature flags. It does not open the camera or contact network services. Filesystem metadata errors are reported as warnings rather than crashing the diagnostic command.
