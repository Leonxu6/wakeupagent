# Runtime boundary audits

WakeUpAgent runs inside a long-lived process that owns camera, model, messaging, browser, and local automation resources. A small call that blocks forever, replaces the process, launches an interactive child, or deserializes executable objects can therefore break components far away from the original call site.

The runtime-global-state advisory composes focused checks for four families:

- **Unbounded waits:** `concurrent.futures.wait`, `as_completed`, `asyncio.wait`, and `select.select` should declare finite timeout behavior where supported.
- **Process and user-visible side effects:** direct `os.fork`, `os.forkpty`, `os.exec*`, `os.spawn*`, `pty.spawn`, `signal.pause`, and `webbrowser.open*` calls are surfaced for explicit lifecycle review.
- **Executable or implementation-specific deserialization:** `marshal`, pandas pickle, dill, cloudpickle, NumPy pickle-enabled loads, and `torch.load(..., weights_only=False)` create deliberate trust boundaries.
- **Native/runtime hooks:** dynamic `ctypes` library loads, SQLite extension loading, `runpy` execution, and `subprocess.Popen(preexec_fn=...)` execute code outside ordinary Python call boundaries.

The checks remain **advisory** so legitimate platform integrations do not make main permanently red. Reviewers should require a clear answer for trust source, lifetime or timeout, cancellation behavior, privacy impact, and cleanup before accepting a new finding.

Each rule has an isolated regression test. The goal is precise maintenance signal, not a broad grep that rewards false positives.
