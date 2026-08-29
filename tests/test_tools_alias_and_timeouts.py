import subprocess

import tools


def test_contact_alias_resolution_is_normalized_without_ambiguity():
    contacts = {"mentor": "Dr Xu", "family": "Mom", "Ａlice": "Alice Chen"}
    assert tools._resolve_contact_alias("MENTOR", contacts) == "Dr Xu"
    assert tools._resolve_contact_alias("Alice", contacts) == "Alice Chen"
    assert tools._resolve_contact_alias("unknown", contacts) is None
    assert tools._resolve_contact_alias("mentor", {"mentor": "A", "MENTOR": "B"}) is None
    assert tools._resolve_contact_alias("Alice", {"Ａlice": "A", "Alice": "B"}) is None
    assert tools._resolve_contact_alias("mentor", []) is None


def test_force_close_app_reports_missing_osascript(monkeypatch):
    monkeypatch.setattr(tools, "_feature_enabled", lambda name: True)
    monkeypatch.setattr(
        tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("osascript")),
    )
    result = tools.force_close_app.invoke({"app_name": "Safari"})
    assert "osascript" in result
    assert "macOS" in result


def test_force_close_app_reports_timeouts_distinctly(monkeypatch):
    monkeypatch.setattr(tools, "_feature_enabled", lambda name: True)
    monkeypatch.setattr(
        tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("osascript", 10)),
    )
    result = tools.force_close_app.invoke({"app_name": "Safari"})
    assert "超时" in result
    assert "Safari" in result


def test_wechat_reports_missing_platform_automation_without_contact_leak(monkeypatch):
    import config

    monkeypatch.setattr(tools, "_feature_enabled", lambda name: True)
    monkeypatch.setattr(config, "WECHAT_CONTACTS", {"mentor": "Private Contact"})
    monkeypatch.setattr(
        tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("osascript")),
    )
    result = tools.send_wechat_shame_message.invoke({"target": "MENTOR", "message": "hello"})
    assert "osascript" in result
    assert "Private Contact" not in result
