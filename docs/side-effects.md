# Side-effect boundaries

WakeUpAgent mixes perception with optional desktop automation. The project treats those capabilities as separate trust boundaries.

## Default behavior

Side-effecting capabilities are opt-in. External messaging, process control, local TTS, and browser control remain disabled until their corresponding environment flags are explicitly enabled. Browser navigation also accepts only validated HTTP(S) URLs.

The legacy `chaos_terminal_punishment` symbol remains importable only so old checkpoints can deserialize safely. It is intentionally inert, is not registered in `ALL_TOOLS`, and must not be advertised by prompts or documentation as a supported escalation path. Do not recreate its disruptive terminal/process behavior through combinations of other tools.

## External messaging

Enable `WAKEUP_ALLOW_EXTERNAL_MESSAGING=true` only after reviewing the contact mapping and macOS Accessibility permissions. Targets must resolve through configured aliases. Message text is bounded and control characters are rejected before AppleScript runs.

For a real installation, prefer `WAKEUP_WECHAT_CONTACTS_JSON` in `.env` instead of editing `config.py`, for example `{"family":"Mom","mentor":"Dr Xu"}`. The mapping parser requires a bounded JSON object with clean string keys and values; malformed JSON, padded names, control characters, non-string values, and oversized maps fail during configuration loading. Do not commit real private contact names to the repository.

## Process control

Enable `WAKEUP_ALLOW_PROCESS_CONTROL=true` only on a machine where closing an explicitly named application is acceptable. Application names are validated before subprocess execution. WakeUpAgent intentionally does not use fuzzy `pkill` matching because a partial process-name match can terminate unrelated work.

## TTS and browser control

Enable local TTS or browser control only when foreground interruptions are acceptable. A model decision never overrides these local feature gates. If a tool reports that a capability is disabled, orchestration should accept that result rather than retrying or finding another side-effect path.

## Testing expectations

Tests for side-effecting tools must mock browser and subprocess boundaries. Invalid inputs and disabled feature flags must be proven to return before external calls. Prompts, README examples, and architecture docs must describe only capabilities that are actually registered. Pull requests should call out any new camera, filesystem, browser, contact, subprocess, or secret handling explicitly.
