import json

from diagnostics import Check, checks_payload, format_checks_json


def test_checks_payload_is_json_safe_and_single_line():
    checks = [Check("model\nname", False, "failed\nwith\tdetail")]
    assert checks_payload(checks) == [
        {"name": "model name", "ok": False, "detail": "failed with detail"}
    ]


def test_format_checks_json_round_trips_unicode():
    checks = [Check("模型", True, "已配置")]
    encoded = format_checks_json(checks)
    assert json.loads(encoded) == [{"name": "模型", "ok": True, "detail": "已配置"}]
    assert "模型" in encoded


def test_checks_payload_returns_new_records():
    checks = [Check("python", True, "3.12")]
    first = checks_payload(checks)
    first[0]["detail"] = "changed"
    assert checks_payload(checks)[0]["detail"] == "3.12"
