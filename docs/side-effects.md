# Side-effect boundaries

WakeUpAgent mixes perception with optional desktop automation. The project treats those capabilities as separate trust boundaries.

## Default behavior

Browser navigation accepts only validated HTTP(S) URLs. External messaging and process control are disabled by default. The legacy chaos-terminal action remains importable only for old checkpoint compatibility and is not registered with the agent.

## External messaging

Enable `WAKEUP_ALLOW_EXTERNAL_MESSAGING=true` only after reviewing `WECHAT_CONTACTS` and macOS Accessibility permissions. Targets must resolve through configured aliases. Message text is bounded and control characters are rejected before AppleScript runs.

## Process control

Enable `WAKEUP_ALLOW_PROCESS_CONTROL=true` only on a machine where closing an explicitly named application is acceptable. Application names are validated before subprocess execution. WakeUpAgent intentionally does not use fuzzy `pkill` matching because a partial process-name match can terminate unrelated work.

## Testing expectations

Tests for side-effecting tools must mock browser and subprocess boundaries. Invalid inputs and disabled feature flags must be proven to return before external calls. Pull requests should call out any new camera, filesystem, browser, contact, subprocess, or secret handling explicitly.
