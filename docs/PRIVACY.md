# Privacy

WakeUpAgent is designed so camera frames stay on the local machine. The perception pipeline may send a short text description of an observation to the configured cloud model, but it does not send the raw frame through the graph API.

## Local data

The following data can exist on disk locally:

- `superego.db`: LangGraph checkpoint state and message history.
- `memory/daily_reports.md`: generated daily summaries.
- `.env`: operator secrets such as `DEEPSEEK_API_KEY`.
- MediaPipe model assets in the repository root.

Runtime paths are resolved relative to the project root, so launching the program from another working directory does not silently create extra copies of the checkpoint or reports.

## External boundaries

- DeepSeek receives text supplied to the decision graph.
- Ollama is expected to run locally unless the operator changes its host.
- Browser, TTS, WeChat, and app-control tools are local side effects.
- WeChat and app termination are disabled by default and require explicit flags.

## Development rules

Do not add camera frames, generated reports, real contact names, `.env` files, API keys, or checkpoint databases to tests or documentation. Use synthetic fixtures and mocks when a test needs to cross an external boundary.
