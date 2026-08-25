# Security boundaries

WakeUpAgent intentionally exposes narrow capabilities instead of general system access.

## Untrusted inputs

Treat model output, environment variables, URLs, application names, message text, structured response objects, and external command errors as untrusted. Validate their shape and size before they cross into network, browser, subprocess, or persistence APIs.

## Explicit opt-ins

External messaging and process control are disabled unless explicitly enabled. A new side-effecting tool should follow the same pattern. Do not infer consent from the model response or from another feature flag.

## No generic shell bridge

A generic command-execution tool would collapse several trust boundaries into one. Prefer narrow functions whose arguments can be validated and whose results can be redacted and bounded.

## Secret handling

Secrets belong in environment configuration and should never be printed by diagnostics. Contact aliases and their real destinations are also sensitive local configuration even when they are not traditional credentials.

## Error handling

Return enough detail to diagnose a failure without echoing private input or unlimited output. Broad exception catches should convert only expected boundary failures; programming errors should remain visible during development.
