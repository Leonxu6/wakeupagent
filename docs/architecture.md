# Runtime architecture

WakeUpAgent is intentionally small: perception collects local signals, the graph turns those signals into a decision, and side-effecting tools execute only after validation and feature-gate checks. Keeping these stages separate makes failures easier to diagnose and prevents model output from flowing directly into operating-system actions.

## Main components

- `main.py` owns process startup, command selection, and shutdown.
- `graph.py` orchestrates model calls and decision handling.
- `history.py` keeps a bounded, normalized context window.
- `config.py` assembles validated runtime configuration from `settings.py`.
- `diagnostics.py` performs side-effect-free installation checks.
- `safety.py` contains reusable validators for URLs, text, application names, and bounded numeric values.
- `tools.py` is the trust boundary for browser, TTS, process-control, and external-messaging effects.

## Data flow

1. Startup loads configuration and validates environment-derived values.
2. Perception produces observations without granting them execution authority.
3. Observations are normalized before entering bounded history.
4. The model proposes a decision.
5. Structured decision text is normalized and validated.
6. A tool may run only when its explicit opt-in flag and input validator both allow it.
7. Tool results return bounded, non-sensitive summaries to the graph.

## Trust boundaries

Model output is untrusted text. Environment variables are untrusted configuration. Contact mappings are local secrets. URLs, application names, subprocess arguments, and message bodies cross an operating-system or network boundary and therefore receive stricter validation than ordinary display text.

The project deliberately avoids giving the graph a generic shell tool. Side effects should be represented by narrow functions with explicit validation and bounded behavior.

## Design rule

When adding a feature, keep validation close to the boundary, keep state bounded, and make failure modes observable. Prefer a small new validator or explicit tool over a permissive generic escape hatch.
