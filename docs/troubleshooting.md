# Troubleshooting

Start with the side-effect-free diagnostics command before changing code or permissions. The goal is to distinguish installation problems from model, camera, or tool failures.

## Startup fails immediately

Check the Python version first. WakeUpAgent targets Python 3.12 or newer. Then verify `.env` values do not contain leading/trailing whitespace and that service URLs are complete HTTP(S) URLs. Run the diagnostics command and read the first failing critical check.

## Model assets are missing

Diagnostics expects the pose and gesture model files to exist as regular, non-empty files. Re-download a missing asset instead of creating a placeholder. If the path exists but is a directory or unreadable file, fix the path or file permissions.

## Local model service is unreachable

A syntactically valid `OLLAMA_HOST` only proves configuration shape, not network availability. Confirm the configured host separately and keep the diagnostic command side-effect free. Avoid weakening URL validation just to accept a malformed local address.

## External messaging does nothing

External messaging is opt-in. Confirm the corresponding feature flag is enabled and that the local contact mapping contains the requested alias. Contact names and destinations should not be echoed back into model-visible tool results.

## Process control is refused

Process control is intentionally narrow. Application names must pass validation and the process-control feature flag must be enabled. Do not replace this with arbitrary shell execution.

## Runtime stops during shutdown

Known executor-shutdown races may be suppressed; unrelated `RuntimeError` exceptions should still surface. Capture the exact exception before broadening shutdown handling.

## Context appears truncated

History is bounded by design. Check the configured history limits before increasing them. Large context windows increase memory use and can amplify stale or private information.

## Reporting a bug

Include the command mode, Python version, sanitized diagnostics output, the smallest reproduction, and the relevant stack trace. Remove API keys, contact mappings, private messages, and local user paths before posting logs.
