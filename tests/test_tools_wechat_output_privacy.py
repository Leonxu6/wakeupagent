from types import SimpleNamespace
from unittest.mock import patch

import tools


def test_wechat_success_result_does_not_echo_private_alias(monkeypatch):
    monkeypatch.setattr(tools, "_feature_enabled", lambda name: True)
    monkeypatch.setattr(tools.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""))

    with patch("config.WECHAT_CONTACTS", {"family": "Private Contact"}):
        result = tools.send_wechat_shame_message.invoke({"target": "family", "message": "hello"})

    assert result == "消息发送完成"
    assert "family" not in result
    assert "Private Contact" not in result
