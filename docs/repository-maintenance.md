# Repository maintenance

WakeUpAgent keeps a small set of offline audits beside the unit tests. They protect repository-level contracts that are easy to miss when a focused feature test still passes.

## Run the audit suite

From the repository root:

```bash
uv run python maintenance/run_all.py
uv run python maintenance/safety_docs_audit.py
```

The commands do not start the camera, contact network services, send messages, open applications, or invoke process-control tools. They inspect tracked source, metadata, documentation, workflow text, model-file metadata, and Git's tracked-file index.

## What is checked

The maintenance suite verifies Python syntax without importing runtime modules, UTF-8 text integrity, tracked-file hygiene, maintainer documentation, environment-template parity, privacy-critical `.gitignore` rules, workflow permissions and versioned action refs, side-effect opt-in gates, CLI diagnostics, project metadata, lockfile alignment, documented commands, case-insensitive path collisions, credential-like filenames, pytest naming, and the flat runtime import layout.

The dedicated safety-documentation audit also makes sure the documented opt-in flags match the runtime contract and that the legacy chaos action remains documented as unregistered.

## Adding an audit

Keep new audits deterministic, side-effect-free, bounded in runtime, and useful on a fresh clone. Prefer repository facts over stylistic preferences. Every audit should have focused tests for both a passing repository shape and the failure it is meant to catch.

If an audit depends on Git state, use tracked paths rather than walking ignored runtime data. If it reads text, report compact file-relative errors so CI output stays actionable.

## Failure policy

A maintenance audit failure is a repository contract failure, not a reason to skip the check. Fix the underlying source, configuration, documentation, or workflow drift. Do not weaken an audit merely to recover a green CI run unless the contract itself has intentionally changed and that change is documented and tested.
