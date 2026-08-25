# Installation diagnostics and maintenance audits

WakeUpAgent includes a side-effect-free diagnostics path for checking a local installation before starting the camera, model clients, or automation tools.

## Human-readable installation check

Run:

```bash
uv run main.py --check
```

The command reports one line per check. A non-zero exit code means a required runtime dependency is not usable. Critical checks currently include:

- Python 3.12 or newer.
- The bundled MediaPipe pose and gesture model assets.
- Runtime configuration parsing.
- Writable parent directories for checkpoint and daily-report persistence.
- A configured DeepSeek API key.

Service URLs and opt-in feature flags are also reported. The check validates configuration only; it does not open the camera, contact network services, send messages, launch applications, or invoke other side effects.

## Machine-readable diagnostics

For CI, setup scripts, or local tooling, run:

```bash
uv run main.py --check-json
```

The output is a JSON array. Each entry contains `name`, `ok`, and `detail`. Names are unique after whitespace normalization, details are rendered on one line, and malformed or ambiguous check collections are rejected instead of producing misleading output.

The JSON mode uses the same exit-code policy as the human-readable mode, so callers can consume structured output without losing failure semantics.

## Invalid local configuration

Diagnostics load the validated runtime configuration defensively. If an environment variable has an invalid type, range, URL, path, or other rejected value, the diagnostic command reports a `configuration` failure instead of crashing during module import. This makes configuration mistakes discoverable before the main runtime starts.

Tracked environment templates are audited separately. Secret-like keys such as API keys, tokens, passwords, and generic secrets must stay empty in `.env.example`; real credentials belong only in untracked local configuration.

## Repository maintenance audits

The main CI workflow runs lightweight repository audits before the full test suite:

```bash
uv run python maintenance/ci_contract_audit.py
uv run python maintenance/env_example_audit.py
uv run python maintenance/python_version_audit.py
uv run python maintenance/model_asset_audit.py
```

These checks protect a few maintenance invariants that ordinary unit tests can miss:

- CI must retain `main` push and pull-request triggers, Python provisioning, frozen dependency sync, and test execution.
- `.env.example` must remain syntactically clean and must not contain populated secret-like settings.
- `.python-version` must agree with the `requires-python` floor in `pyproject.toml`.
- Required `.task` model assets must exist as non-empty regular files.

The Markdown link checker remains available as an offline maintainer tool:

```bash
uv run python maintenance/markdown_links.py
```

It checks repository-local targets without making network requests and flags links that resolve outside the repository root.

## Interpreting warnings

Not every warning is fatal. For example, the project is primarily designed for macOS, so a different platform can be reported without changing the exit code by itself. By contrast, missing required models, invalid configuration, unusable persistence paths, or missing required cloud credentials make the installation check fail because the normal runtime cannot operate reliably without them.

When adding a new diagnostic check, keep it deterministic and side-effect-free, give it a stable unique name, and decide explicitly whether failure should block startup readiness.
