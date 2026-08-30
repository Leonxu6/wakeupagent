from maintenance.websocket_timeout_audit import audit_source

def test_websocket_timeout_is_allowed(): assert audit_source("websockets.connect(url, open_timeout=10)\n")==[]
def test_websocket_missing_timeout_is_reported(): assert audit_source("websockets.connect(url)\n")==["websockets.connect() without open_timeout on line 1"]
