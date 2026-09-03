from pathlib import Path

from maintenance.side_effect_contract_audit import audit


def _write_contract(root: Path, *, gated: bool = True, include_chaos: bool = False, graph_claim: str = ""):
    gate = 'if not _feature_enabled("WAKEUP_ALLOW_TTS"):\n        return "disabled"' if gated else 'return "ok"'
    registered = ", chaos_terminal_punishment" if include_chaos else ""
    (root / "tools.py").write_text(
        f'def play_tts_punishment(text):\n    {gate}\n'
        'def send_wechat_shame_message(target, message):\n    if not _feature_enabled("WAKEUP_ALLOW_EXTERNAL_MESSAGING"):\n        return "disabled"\n'
        'def open_webpage(url):\n    if not _feature_enabled("WAKEUP_ALLOW_BROWSER_CONTROL"):\n        return "disabled"\n'
        'def force_close_app(app_name):\n    if not _feature_enabled("WAKEUP_ALLOW_PROCESS_CONTROL"):\n        return "disabled"\n'
        'def chaos_terminal_punishment(message):\n    return "disabled"\n'
        f'ALL_TOOLS = [play_tts_punishment{registered}]\n',
        encoding="utf-8",
    )
    (root / "graph.py").write_text(f'_SYSTEM_PROMPT = {graph_claim!r}\n', encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "side-effects.md").write_text(
        "The legacy `chaos_terminal_punishment` entry point is intentionally inert and is not registered in `ALL_TOOLS`.\n",
        encoding="utf-8",
    )


def test_side_effect_contract_accepts_gated_tools_and_safe_prompt(tmp_path: Path):
    _write_contract(tmp_path)
    assert audit(tmp_path) == []


def test_side_effect_contract_detects_missing_gate_and_chaos_registration(tmp_path: Path):
    _write_contract(tmp_path, gated=False, include_chaos=True)
    failures = audit(tmp_path)
    assert any("WAKEUP_ALLOW_TTS" in item for item in failures)
    assert any("chaos" in item for item in failures)


def test_side_effect_contract_rejects_disruptive_prompt_claims(tmp_path: Path):
    _write_contract(tmp_path, graph_claim="use chaos_terminal_punishment and 摧毁环境")
    failures = audit(tmp_path)
    assert any("graph.py" in item and "chaos_terminal_punishment" in item for item in failures)
    assert any("graph.py" in item and "摧毁环境" in item for item in failures)
