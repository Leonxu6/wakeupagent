# Observability conventions

WakeUpAgent should make failures diagnosable without turning logs into a second copy of private runtime state.

## Log useful facts

Prefer stable event names, component names, elapsed time, bounded counts, and sanitized error summaries. Good examples are `model_request_failed`, `tool_refused`, `diagnostic_check_failed`, and `history_trimmed`.

## Avoid sensitive payloads

Do not log API keys, contact mappings, message bodies, raw camera observations, complete model prompts, or full subprocess output. If an external command fails, keep only a bounded error detail that is sufficient to identify the class of failure.

## Severity

Use informational messages for expected lifecycle events, warnings for recoverable configuration or dependency problems, and errors for failed operations that change user-visible behavior. Validation failures caused by untrusted model output are expected refusals, not crashes.

## Diagnostics versus runtime logs

`diagnostics.py` is a deterministic installation report and must remain side-effect free. Runtime logs can describe transient events, but should not silently change configuration or probe external services just to produce a status line.

## Future metrics

If metrics are added, start with bounded counters and latency histograms around model calls, tool invocations, validation refusals, and shutdown. Never use high-cardinality labels containing user text, URLs, contact aliases, or file paths.