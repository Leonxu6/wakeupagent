# Repository maintenance

WakeUpAgent keeps a small set of offline audits beside the unit tests. They protect repository-level contracts that are easy to miss when a focused feature test still passes.

## Run the audit suite

From the repository root:

```bash
uv run python -m maintenance.run_all
uv run python maintenance/safety_docs_audit.py
```

The commands do not start the camera, contact network services, send messages, open applications, or invoke process-control tools. They inspect tracked source, metadata, documentation, workflow text, model-file metadata, and Git's tracked-file index.

## Blocking and advisory checks

`maintenance.run_all` has two stages. Blocking audits represent repository contracts that are already clean on `main`; a finding makes the command fail. Advisory audits are newer repository-wide rules whose legacy backlog has not yet been fully triaged. Their findings are printed with an `[advisory]` prefix but do not turn an otherwise healthy main branch red.

New broad rules should normally begin as advisories. Review their findings, remove false positives, remediate genuine problems, and add focused regression tests. Promote an advisory into the blocking audit tuple only after the repository is clean for that rule. This keeps CI strict without making a freshly introduced rule hide useful signal behind a wall of known legacy findings.

The current advisory layer covers resource and lifecycle risks such as broad exception suppression, race-prone temporary names, unbounded queues/deques/caches, process-global environment/timezone/numerical state, process limits and timers, missing client timeouts, long-lived HTTP sessions, anonymous asyncio tasks, multiprocessing lifecycle policy, pickle-backed `shelve`, and archive extraction review.

## What is checked

The maintenance suite verifies Python syntax without importing runtime modules, UTF-8 text integrity, tracked-file hygiene, maintainer documentation, environment-template parity, privacy-critical `.gitignore` rules, workflow permissions and versioned action refs, side-effect opt-in gates, CLI diagnostics, project metadata, lockfile alignment, documented commands, case-insensitive path collisions, credential-like filenames, pytest naming, and the flat runtime import layout.

Runtime-specific Python checks are scoped away from tests and maintenance tooling. They reject assertions that disappear under `python -O`, `BaseException` handlers that can swallow shutdown signals, runtime `sys.path` mutation, blocking `input()` prompts, unverified SSL contexts, host-derived UUIDv1 identifiers, and `socket.create_connection()` calls without explicit timeouts. This keeps the long-running agent predictable under unattended execution.

The dedicated safety-documentation audit also makes sure the documented opt-in flags match the runtime contract and that the legacy chaos action remains documented as unregistered.

## Adding an audit

Keep new audits deterministic, side-effect-free, bounded in runtime, and useful on a fresh clone. Prefer repository facts over stylistic preferences. Every audit should have focused tests for both a passing repository shape and the failure it is meant to catch.

If an audit depends on Git state, use tracked paths rather than walking ignored runtime data. If it reads text, report compact file-relative errors so CI output stays actionable.

## Failure policy

A blocking maintenance audit failure is a repository contract failure, not a reason to skip the check. Fix the underlying source, configuration, documentation, or workflow drift. Do not weaken a blocking audit merely to recover a green CI run unless the contract itself has intentionally changed and that change is documented and tested.

An advisory finding is different: it is a tracked maintenance backlog item. Do not hide it, but do not convert it into a blocking failure until the repository can satisfy the rule honestly.
