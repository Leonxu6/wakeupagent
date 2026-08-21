# Safety model

WakeUpAgent can cross from model output into real local side effects, so the safest default is intentionally conservative.

## Default behavior

Browser and text-to-speech helpers may be available for ordinary feedback. Contact automation, app termination, and chaos-style automation are disabled unless the operator explicitly opts in through environment flags. A model must never be able to enable those flags itself.

## Side-effect rules

1. Validate user/model input before invoking an OS or network boundary.
2. Prefer a reversible action over an irreversible one.
3. Never interpolate untrusted text into a shell command.
4. Do not log API keys, contact names, camera frames, or generated private memory.
5. Tests must patch side-effecting functions instead of touching the real desktop.
6. Fail closed when a safety flag is missing, malformed, or false.

## Operator checklist

Before enabling a disruptive action, review the function, verify its targets, run the related unit tests, and confirm the setting is documented in `.env.example`. Keep the feature disabled on shared machines and CI runners.
