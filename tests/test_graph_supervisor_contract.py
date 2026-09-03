import graph


def test_system_prompt_does_not_advertise_removed_chaos_mode():
    lowered = graph._SYSTEM_PROMPT.lower()
    assert "50 个终端" not in graph._SYSTEM_PROMPT
    assert "摧毁环境" not in graph._SYSTEM_PROMPT
    assert "人身攻击" in graph._SYSTEM_PROMPT
    assert "legacy chaos mode" in lowered
    assert "不得请求" in graph._SYSTEM_PROMPT


def test_system_prompt_respects_side_effect_feature_gates():
    assert "显式启用" in graph._SYSTEM_PROMPT
    assert "不要尝试绕过" in graph._SYSTEM_PROMPT
    assert "最小影响" in graph._SYSTEM_PROMPT


def test_safe_error_detail_never_echoes_backend_message():
    detail = graph._safe_error_detail(RuntimeError("api-key=super-secret"))
    assert detail == "RuntimeError"
    assert "super-secret" not in detail
