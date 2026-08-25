from pathlib import Path

from maintenance.safety_docs_audit import audit


def test_safety_docs_accept_opt_in_and_disabled_legacy_behavior(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    docs.joinpath("side-effects.md").write_text(
        "WAKEUP_ALLOW_TTS WAKEUP_ALLOW_BROWSER_CONTROL WAKEUP_ALLOW_EXTERNAL_MESSAGING "
        "WAKEUP_ALLOW_PROCESS_CONTROL legacy chaos is not registered\n",
        encoding="utf-8",
    )
    assert audit(tmp_path) == []


def test_safety_docs_report_missing_boundaries(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    docs.joinpath("side-effects.md").write_text("partial\n", encoding="utf-8")
    failures = audit(tmp_path)
    assert any("WAKEUP_ALLOW_TTS" in item for item in failures)
    assert any("legacy chaos" in item for item in failures)
