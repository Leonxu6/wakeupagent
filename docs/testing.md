# Testing guide

The test suite should make unsafe regressions noisy and ordinary refactors cheap.

## Fast local loop

Run the full suite with `uv run pytest -q`. During focused work, run the smallest relevant file first, for example `uv run pytest -q tests/test_safety.py`, then finish with the full suite before merging.

## What deserves a regression test

Add a regression test when a bug can cross a trust boundary, corrupt persistent state, leak private data, hang the runtime, or turn a configuration typo into surprising behavior. Boundary tests are especially valuable for environment parsing, URLs, subprocess inputs, model response shapes, file-system races, and opt-in side effects.

## Test structure

Prefer one behavior per test. Name tests after the contract, not the implementation. Keep external effects mocked at the boundary rather than mocking every internal function. When testing validation, assert that failure happens before the subprocess, browser, network client, or persistence layer is touched.

## Useful categories

- pure validation tests for `safety.py` and `settings.py`;
- state and truncation tests for `history.py`;
- side-effect-free installation checks for `diagnostics.py`;
- graph response-shape and shutdown-race tests;
- tool tests that verify explicit opt-in and redacted results;
- CLI tests that ensure diagnostic modes do not initialize cameras or network clients.

## CI expectations

CI is expected to compile the maintained Python modules and run pytest on Python 3.12. A green local focused test is not enough if the full suite fails. Do not bypass a flaky failure without identifying whether it is a real race, an environment assumption, or a nondeterministic test.

## Adding fixtures

Keep fixtures small and synthetic. Do not commit real contacts, API keys, personal messages, or machine-specific paths. Prefer temporary directories and in-memory fakes for persistence and process boundaries.
