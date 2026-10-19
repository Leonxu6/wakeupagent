import json

import pytest

from diagnostics import Check, checks_payload, checks_report_payload, format_checks_json, format_checks_json_report


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


def test_json_report_summarizes_checks_and_readiness():
    checks = [
        Check("python", True, "3.12"),
        Check("platform", False, "Linux"),
        Check("deepseek-key", False, "optional"),
    ]
    report = checks_report_payload(checks)
    assert report == {
        "schema_version": 1,
        "ready": True,
        "summary": {"total": 3, "ok": 1, "warnings": 2},
        "checks": [
            {"name": "python", "ok": True, "detail": "3.12"},
            {"name": "platform", "ok": False, "detail": "Linux"},
            {"name": "deepseek-key", "ok": False, "detail": "optional"},
        ],
    }
    assert json.loads(format_checks_json_report(checks)) == report


def test_json_report_marks_critical_failure_not_ready():
    report = checks_report_payload([Check("pose-model", False, "missing")])
    assert report["ready"] is False


def test_checks_payload_returns_new_records():
    checks = [Check("python", True, "3.12")]
    first = checks_payload(checks)
    first[0]["detail"] = "changed"
    assert checks_payload(checks)[0]["detail"] == "3.12"


def test_checks_payload_rejects_duplicate_normalized_names():
    with pytest.raises(ValueError, match="duplicate diagnostic check name"):
        checks_payload([Check("model name", True, "ok"), Check("model\nname", False, "bad")])


@pytest.mark.parametrize(
    "checks",
    [
        "not-a-list",
        [object()],
        [Check("", True, "empty name")],
        [Check("python", 1, "non-boolean status")],
    ],
)
def test_checks_payload_rejects_malformed_check_collections(checks):
    with pytest.raises(ValueError):
        checks_payload(checks)
