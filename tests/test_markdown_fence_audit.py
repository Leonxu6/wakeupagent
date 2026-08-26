from maintenance.markdown_fence_audit import audit_text


def test_markdown_fence_audit_accepts_balanced_blocks():
    assert audit_text("```python\nprint('ok')\n```\n") == []


def test_markdown_fence_audit_reports_unclosed_block():
    assert audit_text("intro\n```python\nprint('oops')\n") == [
        "unclosed Markdown fence opened on line 2"
    ]
