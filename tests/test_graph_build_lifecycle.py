import pytest

import graph


class _Connection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_graph_build_closes_checkpoint_connection_when_compile_fails(monkeypatch):
    connection = _Connection()
    monkeypatch.setattr(graph, "_connect_checkpoint", lambda _path: connection)

    def fail_compile(_builder, **_kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(graph.StateGraph, "compile", fail_compile)

    with pytest.raises(RuntimeError, match="compile failed"):
        graph.build_graph()

    assert connection.closed is True
