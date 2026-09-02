import pytest

from maintenance import runtime_global_state_audit as audit


def test_faulthandler_mutations_are_reported():
    source = """
import faulthandler
faulthandler.enable()
faulthandler.register(12)
faulthandler.unregister(12)
faulthandler.disable()
"""

    findings = audit.findings_for_source(source, path="agent.py")

    assert len(findings) == 4
    assert all(item.startswith("agent.py:") for item in findings)
    assert any("faulthandler.enable" in item for item in findings)
    assert any("faulthandler.register" in item for item in findings)
    assert any("faulthandler.unregister" in item for item in findings)
    assert any("faulthandler.disable" in item for item in findings)


def test_unrelated_local_calls_are_ignored():
    assert audit.findings_for_source("value.append(1)\n") == []


def test_source_and_path_contracts_are_explicit():
    with pytest.raises(ValueError, match="source"):
        audit.findings_for_source(None)  # type: ignore[arg-type]
    for path in ("", " agent.py", "agent.py "):
        with pytest.raises(ValueError, match="path"):
            audit.findings_for_source("pass\n", path=path)
