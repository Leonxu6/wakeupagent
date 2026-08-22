# Local development

Use a clean Python 3.12 environment and keep runtime secrets outside the repository.

## Setup

Create the environment with the project package manager, copy `.env.example` to a local `.env` only when needed, and keep that file untracked. Run the side-effect-free installation check before starting camera or messaging features.

## Development loop

1. Reproduce the behavior with the smallest focused test.
2. Change the narrowest component that owns the contract.
3. Run the focused test file.
4. Run `uv run pytest -q` and the maintained compile checks.
5. Run diagnostics when configuration or model paths changed.
6. Review the diff for secrets, local paths, debug prints, and overly broad exception handling.

## Side-effecting features

Keep external messaging and process control disabled while developing unrelated code. Tests should monkeypatch those boundaries rather than interacting with real applications or contacts.

## Commits

A commit should describe one coherent behavior change and include the corresponding validation or test when practical. Avoid mixing formatting churn with functional changes. Documentation should be updated when a setting, safety boundary, or user-visible command changes.