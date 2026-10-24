# Privacy and local data

WakeUpAgent handles observations, model prompts, contact aliases, and potentially sensitive local context. Maintenance should minimize how much of that data leaves the process or is retained.

## Principles

- Treat camera-derived observations and conversation context as private by default.
- Keep history bounded and normalize it before reuse.
- Never commit `.env` files, contact mappings, API keys, transcripts, screenshots, or machine-specific logs.
- Tool return values should contain only what the graph needs to continue, not private destinations or full subprocess output.
- Diagnostics should report configuration state without printing secrets.

## Contacts and messaging

Contact aliases are a local trust boundary. The model may request an alias, but the mapping from alias to real destination stays outside model-visible output. If a message is sent, return a generic success/failure summary rather than the resolved contact name or account identifier.

The WeChat automation program is delivered to `osascript` over standard input. Resolved contacts and message bodies are never placed in process arguments, so they are not exposed through ordinary process-list inspection. They still exist briefly in the local subprocess input and clipboard and therefore remain protected data.

## Logs

When adding logs, prefer event type, bounded error class, and sanitized detail. Avoid dumping raw model payloads, environment dictionaries, command output, or OS paths. Error messages from external programs should be length-bounded before they are returned to the graph.

Daily Markdown reports are append-only but bounded to 5 MiB. A retry for a date already present is idempotent, and a new entry fails before writing when it would exceed the file budget. Archive or rotate the report deliberately before restarting if this limit is reached; do not silently discard older private history.

SQLite checkpoints can contain camera-derived descriptions, model messages, and tool results. Graph startup therefore opens only a regular non-symlink target and narrows the database mode to at most `0600` through its file descriptor, including when reusing an older broadly readable file. Permission hardening must succeed before the graph begins processing observations.

## Development fixtures

Tests must use synthetic names, URLs, messages, and temporary paths. A bug report should be reproducible without real personal data.
