from maintenance.subprocess_shell_audit import audit_source


def test_subprocess_shell_audit_accepts_argument_lists():
    assert audit_source('subprocess.run(["echo", "ok"], check=True)\n') == []


def test_subprocess_shell_audit_reports_shell_true():
    assert audit_source('subprocess.run("echo ok", shell=True)\n') == [
        "subprocess shell=True on line 1"
    ]
