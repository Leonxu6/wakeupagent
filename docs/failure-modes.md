# Failure modes

This document records failures that deserve explicit handling rather than broad exception suppression.

## Configuration

Malformed booleans, numbers, URLs, paths, secrets, and JSON contact mappings should fail at startup with stable messages. Defaults are validated too, because a bad default is still a programming error.

## Perception and model responses

A malformed graph message batch or structured response should be normalized or rejected without turning arbitrary objects into executable instructions. Context should remain bounded even when upstream text is unexpectedly large.

## File system

Model and persistence paths can disappear or change type between checks. Diagnostics should report these races and remain side-effect free rather than creating directories or placeholder files.

## Subprocesses and tools

Tool inputs are validated before subprocess launch. Non-zero exits return bounded error details. External messaging and process control remain disabled unless explicitly enabled.

## Shutdown

Only known executor-shutdown races should be suppressed. Unrelated runtime errors should surface so that genuine bugs are not mistaken for normal teardown.

## Privacy

A technically successful tool call can still be a failure if its result leaks contact names, destinations, message content, or unbounded subprocess output back to the model. Result redaction is part of correctness.
