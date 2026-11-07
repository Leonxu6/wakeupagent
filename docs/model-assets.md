# Model asset lifecycle

WakeUpAgent depends on local pose and gesture model assets. These files are runtime dependencies, not generated placeholders.

## Expected properties

Each configured model path should name a regular, readable, non-empty file.
Symbolic links are rejected so a configured redirect outside the installation
cannot be reported as a ready local model asset. The side-effect-free
diagnostics command checks these properties before the camera pipeline is
started. On platforms with `O_NOFOLLOW`, diagnostics also opens the asset with
link following disabled and derives its type and size from that descriptor.
This closes the window where a regular file could be replaced by a symbolic
link between the initial metadata check and the readiness decision.

Normal perception startup reads each asset through the validated descriptor
before opening the camera, with a 64 MiB per-file budget and a second metadata
check after the read. MediaPipe receives that in-memory snapshot rather than
reopening the pathname later. A path replacement between validation and
detector creation therefore cannot redirect the camera session to different
model bytes, and a growing or unexpectedly large asset fails before camera
activation. Running the main entry point directly retains these guarantees even
when the diagnostics command was skipped.

## Updating an asset

When replacing a model, record the upstream source and version in the change description, confirm licensing permits redistribution, and verify the application still starts on the supported Python/runtime stack. Avoid silently swapping a binary while keeping the same filename if its input or output contract changes.

## Repository hygiene

Do not commit temporary downloads, partial files, local cache directories, or decompressed training artifacts. Large generated assets should not be duplicated under multiple names. If a future model becomes too large for normal Git history, document the distribution mechanism rather than committing fragmented archives.

## Failure diagnosis

A missing model, directory or symbolic link in place of a model, zero-byte file,
or unreadable file should fail diagnostics with a clear local message. Network
access should not be required merely to determine that the installation is
incomplete.
