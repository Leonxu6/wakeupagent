# Failure modes

This document records failures that deserve explicit handling rather than broad exception suppression.

## Configuration

Malformed booleans, numbers, URLs, paths, secrets, and JSON contact mappings should fail at startup with stable messages. Defaults are validated too, because a bad default is still a programming error.

## Perception and model responses

A malformed graph message batch or structured response should be normalized or rejected without turning arbitrary objects into executable instructions. Context should remain bounded even when upstream text is unexpectedly large.

Local model descriptions and classifier context are truncated before whitespace
or control-character normalization. This bounds cleanup work even when a model
backend returns an unexpectedly large response.

On-screen labels also hard-wrap individual tokens to the calculated display
budget. A malformed or code-like model response with no spaces therefore cannot
create a single oversized overlay that extends beyond the camera frame.
Wrapped labels are also capped to the number of lines the current frame can
display. Clipped descriptions end with an ellipsis, so a long but otherwise
valid model response cannot create a text block taller than the camera image.

Memory compaction summarizes the oldest bounded batch and removes only messages included in a successful summary. Model errors or empty summary responses leave the raw history intact for a later retry, preventing silent context loss during an outage.

Tool proposals are executable only when every call names a registered tool, carries a unique normalized call ID, and contains a strict-JSON argument object within the 20,000-byte envelope budget. Structural validation is iterative rather than recursive and also caps nesting at 32 levels and the decoded shape at 2,000 nodes. Unknown tools, duplicate IDs, missing IDs, non-object arguments, cycles, excessive nesting, excessive structural complexity, non-finite numbers, and oversized payloads are rejected before the response is stored or routed to `ToolNode`.

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

Each perception start also receives a monotonically increasing session generation.
Background model work must present the same generation before it can publish a
callback. This prevents a slow analysis thread from a stopped camera session from
injecting a stale observation after the user restarts perception and the shared
stop signal has been cleared for the new session.

The capture loop and teardown path use the same generation ownership. A replaced
loop stops reading its old camera, but its `finally` block may release only that
camera handle; it cannot set the shared stop signal, clear the current frame, or
destroy windows owned by the newer session. This keeps rapid camera restarts from
being cancelled by delayed cleanup from the preceding run.

On-demand camera observations also capture the active generation before waiting
for a fresh frame. They verify that ownership both before and after local model
inference. If perception stops or restarts during a slow description, the tool
returns a cancellation marker and never publishes the stale frame description
into the newer agent session.

## Privacy

A technically successful tool call can still be a failure if its result leaks contact names, destinations, message content, or unbounded subprocess output back to the model. Result redaction is part of correctness.
