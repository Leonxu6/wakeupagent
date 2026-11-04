# Model asset lifecycle

WakeUpAgent depends on local pose and gesture model assets. These files are runtime dependencies, not generated placeholders.

## Expected properties

Each configured model path should name a regular, readable, non-empty file.
Symbolic links are rejected so a configured redirect outside the installation
cannot be reported as a ready local model asset. The side-effect-free
diagnostics command checks these properties before the camera pipeline is
started.

## Updating an asset

When replacing a model, record the upstream source and version in the change description, confirm licensing permits redistribution, and verify the application still starts on the supported Python/runtime stack. Avoid silently swapping a binary while keeping the same filename if its input or output contract changes.

## Repository hygiene

Do not commit temporary downloads, partial files, local cache directories, or decompressed training artifacts. Large generated assets should not be duplicated under multiple names. If a future model becomes too large for normal Git history, document the distribution mechanism rather than committing fragmented archives.

## Failure diagnosis

A missing model, directory or symbolic link in place of a model, zero-byte file,
or unreadable file should fail diagnostics with a clear local message. Network
access should not be required merely to determine that the installation is
incomplete.
