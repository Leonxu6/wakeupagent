import tools


def test_wechat_script_restores_clipboard_on_success_and_error_paths():
    script = tools._wechat_script('Private "Contact"', "secret\\message")
    assert "set oldClipboard to the clipboard" in script
    assert script.count("set the clipboard to oldClipboard") == 2
    assert f"with timeout of {tools._WECHAT_AUTOMATION_TIMEOUT_SEC} seconds" in script
    assert tools._WECHAT_AUTOMATION_TIMEOUT_SEC < tools._WECHAT_PROCESS_TIMEOUT_SEC
    assert "on error errMessage number errNumber" in script
    assert "error errMessage number errNumber" in script
    assert 'Private \\"Contact\\"' in script
    assert "secret\\\\message" in script
