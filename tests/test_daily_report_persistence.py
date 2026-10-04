import os
import stat

import pytest

import graph


def test_daily_report_is_bounded_normalized_and_private(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "daily.md"
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(path))

    graph._save_daily_report("  focused\n\u202e tomorrow  ", "2026-10-03")

    assert path.read_text(encoding="utf-8") == "\n## 2026-10-03\nfocused tomorrow\n"
    assert path.stat().st_mode & 0o077 == 0


def test_daily_report_retries_short_writes_until_entry_is_complete(tmp_path, monkeypatch):
    path = tmp_path / "daily.md"
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(path))
    original_write = graph.os.write
    calls = []

    def short_write(fd, payload):
        calls.append(len(payload))
        return original_write(fd, payload[: max(1, len(payload) // 2)])

    monkeypatch.setattr(graph.os, "write", short_write)
    graph._save_daily_report("complete entry", "2026-10-03")

    assert len(calls) > 1
    assert path.read_text(encoding="utf-8") == "\n## 2026-10-03\ncomplete entry\n"


@pytest.mark.parametrize("date_str", ["2026-1-03", "2026-02-30", "2026-10-03\n", None])
def test_daily_report_rejects_noncanonical_dates(tmp_path, monkeypatch, date_str):
    path = tmp_path / "daily.md"
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(path))

    with pytest.raises(ValueError, match="canonical ISO date"):
        graph._save_daily_report("report", date_str)

    assert not path.exists()


def test_daily_report_rejects_empty_or_non_text_content(tmp_path, monkeypatch):
    path = tmp_path / "daily.md"
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(path))

    for report in ("   ", None, ["structured"]):
        with pytest.raises(ValueError, match="must not be empty"):
            graph._save_daily_report(report, "2026-10-03")

    assert not path.exists()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="platform has no O_NOFOLLOW")
def test_daily_report_refuses_symlink_targets(tmp_path, monkeypatch):
    target = tmp_path / "private.md"
    target.write_text("unchanged", encoding="utf-8")
    link = tmp_path / "daily.md"
    link.symlink_to(target)
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(link))

    with pytest.raises(OSError):
        graph._save_daily_report("report", "2026-10-03")

    assert target.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="platform has no descriptor chmod")
def test_daily_report_tightens_existing_file_permissions(tmp_path, monkeypatch):
    path = tmp_path / "daily.md"
    path.write_text("existing\n", encoding="utf-8")
    path.chmod(0o666)
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(path))

    graph._save_daily_report("private entry", "2026-10-04")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text(encoding="utf-8").endswith("\n## 2026-10-04\nprivate entry\n")


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="platform cannot create and reject FIFOs without blocking",
)
def test_daily_report_refuses_fifo_targets(tmp_path, monkeypatch):
    path = tmp_path / "daily.md"
    os.mkfifo(path)
    monkeypatch.setattr(graph, "DAILY_REPORT_PATH", str(path))

    with pytest.raises(OSError):
        graph._save_daily_report("report", "2026-10-04")
