import tools


def test_process_control_result_hides_backend_exception(monkeypatch):
    monkeypatch.setattr(tools, "_feature_enabled", lambda name: True)

    def fail(*args, **kwargs):
        raise RuntimeError("/Users/private/operator/path")

    monkeypatch.setattr(tools.subprocess, "run", fail)
    result = tools.force_close_app.invoke({"app_name": "Notes"})

    assert result == "Error: 无法请求 Notes 退出"
    assert "/Users/private" not in result
