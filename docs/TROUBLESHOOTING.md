# Troubleshooting

Start with the offline diagnostic:

```bash
uv run main.py --doctor
```

## Missing MediaPipe models

If the doctor reports either `.task` file missing, restore `pose_landmarker_lite.task` and `gesture_recognizer.task` in the repository root. The perception loop intentionally refuses to start without both assets.

## Camera cannot open

Close other applications using the camera and confirm `CAMERA_INDEX` in `config.py`. External webcams commonly use index `1` or `2`. macOS camera permission must be granted to the terminal or editor launching the process.

## Ollama unavailable

Confirm Ollama is running and the configured models are installed:

```bash
ollama list
```

The default local models are `moondream` and `qwen2.5:1.5b`.

## Cloud decision errors

Verify `.env` contains `DEEPSEEK_API_KEY` and that the key is not surrounded by quotes or accidental whitespace. Never paste the key into an issue or CI log.

## Desktop action says disabled

This is the expected fail-closed behavior. WeChat and app termination are not available unless their dedicated environment flags are explicitly true before process startup. Review `docs/SAFETY.md` and `.env.example` before enabling them.

## Tests fail on non-macOS CI

Normal unit tests must mock macOS side effects. A test that invokes `say`, `osascript`, a real browser, a real camera, or a real desktop process is an integration test and should not run in the default CI job.
