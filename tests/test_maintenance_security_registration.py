from maintenance.run_all import _AUDITS


def test_security_and_syntax_audits_are_registered():
    expected = {
        "json_syntax_audit.py",
        "toml_syntax_audit.py",
        "unicode_bidi_audit.py",
        "dynamic_code_audit.py",
        "subprocess_shell_audit.py",
        "unsafe_deserialization_audit.py",
        "weak_hash_audit.py",
        "tls_verification_audit.py",
        "tempfile_safety_audit.py",
        "dependency_source_audit.py",
        "markdown_fence_audit.py",
    }
    assert expected.issubset(set(_AUDITS))


def test_reliability_and_async_audits_are_registered():
    expected = {
        "bare_except_audit.py",
        "silent_exception_audit.py",
        "mutable_default_audit.py",
        "wildcard_import_audit.py",
        "debug_call_audit.py",
        "os_system_audit.py",
        "datetime_utcnow_audit.py",
        "absolute_user_path_audit.py",
        "duplicate_definition_audit.py",
        "unsafe_chmod_audit.py",
        "http_timeout_audit.py",
        "urlopen_timeout_audit.py",
        "subprocess_timeout_audit.py",
        "async_blocking_sleep_audit.py",
        "async_subprocess_audit.py",
    }
    assert expected.issubset(set(_AUDITS))
