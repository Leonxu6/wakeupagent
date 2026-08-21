# Security Policy

WakeUpAgent can interact with a camera, browser, text-to-speech, and macOS automation APIs. Treat changes to those boundaries as security-sensitive.

## Supported version

Security fixes are applied to the `main` branch while the project is pre-1.0.

## Reporting a vulnerability

Please report vulnerabilities privately to the repository owner rather than posting credentials, personal data, or exploit details in a public issue. Include the affected component, reproduction conditions, expected impact, and a minimal safe proof of concept when possible.

## Security principles

- Side-effecting actions should be opt-in and easy to disable.
- Inputs crossing into shells, AppleScript, URLs, subprocesses, or model prompts must be validated.
- Secrets belong in environment variables and must never be printed in logs.
- Camera frames and generated memory should remain local unless a feature explicitly documents otherwise.
- Tests should mock external side effects.
