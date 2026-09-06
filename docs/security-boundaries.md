# Security boundaries

WakeUpAgent intentionally exposes narrow capabilities instead of general system access.

## Untrusted inputs

Treat model output, environment variables, URLs, application names, message text, structured response objects, and external command errors as untrusted. Validate their shape and size before they cross into network, browser, subprocess, or persistence APIs.

## Model text boundary

Model and orchestration text should pass through the shared `text_safety` helpers before it is logged, persisted, summarized, or reused as context. The boundary removes control, bidirectional, and invisible formatting characters, replaces surrogate code points that cannot be safely encoded, and applies explicit character budgets.

Both output size and raw normalization work are bounded. Extremely large raw strings are inspected only up to the shared raw-input ceiling, so asking for a small normalized result cannot force the agent to walk an arbitrarily large model or environment value. Structured content is consumed incrementally up to its block budget rather than copied with a broad slice.

Structured model responses are not generic dictionaries. Only recognized text block types (`text`, `input_text`, `output_text`, plus legacy blocks without a type) may contribute text. Image, tool-result, and other non-text blocks must not be folded into prompts or persistent summaries merely because they happen to contain a `text` field.

## Persistence paths

Checkpoint and report paths are configuration boundaries, not free-form display text. Diagnostics reject padded paths and Unicode control, format, or surrogate characters before path expansion/resolution. This prevents invisible path spelling differences from reaching persistence checks and keeps diagnostic output stable across terminals and filesystems.

## Explicit opt-ins

External messaging and process control are disabled unless explicitly enabled. A new side-effecting tool should follow the same pattern. Do not infer consent from the model response or from another feature flag.

## No generic shell bridge

A generic command-execution tool would collapse several trust boundaries into one. Prefer narrow functions whose arguments can be validated and whose results can be redacted and bounded.

## Secret handling

Secrets belong in environment configuration and should never be printed by diagnostics. Contact aliases and their real destinations are also sensitive local configuration even when they are not traditional credentials.

## Error handling

Return enough detail to diagnose a failure without echoing private input or unlimited output. Broad exception catches should convert only expected boundary failures; programming errors should remain visible during development.
