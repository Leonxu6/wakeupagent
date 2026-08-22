# Release process

WakeUpAgent does not need frequent version bumps to look active. A release should correspond to a coherent user-visible milestone.

## Before a release

Run the full test suite on the supported Python version, run side-effect-free diagnostics on a clean checkout, review configuration changes against `.env.example`, and verify new side effects remain explicit opt-ins. Check documentation for renamed settings and removed behavior.

## Release notes

Summarize user-visible features, behavior changes, configuration migrations, fixed failure modes, and known limitations. Do not list every internal commit. Highlight changes that affect privacy, external messaging, process control, model assets, or minimum runtime versions.

## Compatibility

If a configuration key changes, provide a migration note. If a model or persistence format changes, document whether existing local state remains readable. Breaking changes should be deliberate rather than an accidental consequence of refactoring.

## After release

Verify the repository default branch remains green, links in the README still point to current docs, and the installation path works from a fresh environment. Avoid empty releases or version bumps whose only purpose is activity.