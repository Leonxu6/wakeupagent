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
