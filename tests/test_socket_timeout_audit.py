from maintenance.socket_timeout_audit import audit_source


def test_socket_timeout_audit_allows_bounded_connections():
    assert audit_source("import socket\nsocket.create_connection(('host', 443), timeout=5)\n") == []


def test_socket_timeout_audit_reports_unbounded_connections():
    assert audit_source("import socket\nsocket.create_connection(('host', 443))\n") == [
        "socket.create_connection() without timeout on line 2"
    ]
