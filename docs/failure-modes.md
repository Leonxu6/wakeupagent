# Failure modes

This document records failures that deserve explicit handling rather than broad exception suppression.

## Configuration

Malformed booleans, numbers, URLs, paths, secrets, and JSON contact mappings should fail at startup with stable messages. Defaults are validated too, because a bad default is still a programming error.

## Perception and model responses

A malformed graph message batch or structured response should be normalized or rejected without turning arbitrary objects into executable instructions. Context should remain bounded even when upstream text is unexpectedly large.

Local model descriptions and classifier context are truncated before whitespace
or control-character normalization. This bounds cleanup work even when a model
backend returns an unexpectedly large response.

Memory compaction summarizes the oldest bounded batch and removes only messages included in a successful summary. Model errors or empty summary responses leave the raw history intact for a later retry, preventing silent context loss during an outage.

Tool proposals are executable only when every call names a registered tool, carries a unique normalized call ID, and contains a strict-JSON argument object within the 20,000-byte envelope budget. Unknown tools, duplicate IDs, missing IDs, non-object arguments, non-finite numbers, and oversized payloads are rejected before the response is stored or routed to `ToolNode`.

## File system

Model and persistence paths can disappear or change type between checks. Diagnostics should report these races and remain side-effect free rather than creating directories or placeholder files.

## Subprocesses and tools

Tool inputs are validated before subprocess launch. Non-zero exits return bounded error details. External messaging and process control remain disabled unless explicitly enabled.

## Shutdown

Only known executor-shutdown races should be suppressed. Unrelated runtime errors should surface so that genuine bugs are not mistaken for normal teardown.

Perception startup is terminal when a required MediaPipe model is missing: the
session clears any cached frame and keeps the shared stop signal set. Camera
observation tools therefore cancel immediately instead of waiting for a capture
from a session that never started.

## Privacy

A technically successful tool call can still be a failure if its result leaks contact names, destinations, message content, or unbounded subprocess output back to the model. Result redaction is part of correctness.
