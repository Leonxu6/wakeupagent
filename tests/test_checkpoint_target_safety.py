import os

import pytest

import graph


def test_checkpoint_target_accepts_missing_and_regular_files(tmp_path):
    missing = tmp_path / "missing.db"
    graph._validate_checkpoint_target(missing)

    regular = tmp_path / "state.db"
    regular.write_bytes(b"")
    graph._validate_checkpoint_target(regular)


def test_checkpoint_target_rejects_symbolic_links(tmp_path):
    target = tmp_path / "real.db"
    target.write_bytes(b"")
    link = tmp_path / "state.db"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        graph._validate_checkpoint_target(link)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable on this platform")
def test_checkpoint_target_rejects_special_files_without_opening_them(tmp_path):
    fifo = tmp_path / "state.db"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        graph._validate_checkpoint_target(fifo)


def test_checkpoint_connection_closes_when_post_open_validation_fails(monkeypatch, tmp_path):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    calls = 0

    def validate(_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("target changed while opening")

    monkeypatch.setattr(graph, "_validate_checkpoint_target", validate)
    monkeypatch.setattr(graph.sqlite3, "connect", lambda *args, **kwargs: connection)

    with pytest.raises(ValueError, match="target changed"):
        graph._connect_checkpoint(tmp_path / "state.db")

    assert connection.closed
