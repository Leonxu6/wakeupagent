# Maintainer review checklist

Use this checklist for changes that affect runtime behavior.

## Correctness

- The change has a reproducible reason to exist.
- Boundary inputs have explicit type, size, and shape rules.
- Failure behavior is deterministic and does not hide unrelated exceptions.
- State growth is bounded.

## Safety and privacy

- Side effects remain explicit opt-ins.
- No generic shell or arbitrary process execution was introduced.
- Logs and tool results do not expose secrets, contacts, private message text, or unbounded command output.
- URLs, subprocess arguments, application names, and messages are validated before use.

## Verification

- Focused regression tests cover the new contract.
- The full test suite passes.
- Diagnostics remain side-effect free.
- `.env.example` and configuration docs match runtime settings.
- New files do not contain local paths, credentials, caches, or generated artifacts.

## Documentation

Update user-facing docs when a command, environment variable, runtime requirement, model asset, or safety boundary changes. Prefer durable explanations over comments that merely restate code.
