# Contributing

WakeUpAgent is an experimental edge-first productivity agent. Contributions should improve reliability, privacy, safety, testability, or developer experience without making local side effects harder to understand or disable.

## Development setup

1. Install Python 3.12 and `uv`.
2. Run `uv sync --frozen`.
3. Run `uv run --with pytest pytest -q` before opening a pull request.
4. Keep changes small enough to review and add regression coverage for behavior changes.

## Project expectations

- Keep camera processing and local automation explicit and user-controlled.
- Do not commit API keys, contact names, personal images, checkpoints, or generated reports.
- New side-effecting tools must be disabled by default or clearly gated by configuration.
- Prefer deterministic helper functions that can be tested without camera, browser, WeChat, or macOS UI access.
- Error messages should explain the failed operation without leaking secrets.

## Commit style

Use focused messages such as `fix(perception): ...`, `test(tools): ...`, `docs: ...`, or `ci: ...`. Avoid formatting-only commits unless formatting is itself the maintenance task.
