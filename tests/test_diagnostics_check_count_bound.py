import pytest

from diagnostics import Check, checks_payload


def test_diagnostic_rendering_rejects_excessive_check_collections():
    checks = [Check(f"check-{index}", True, "ok") for index in range(1001)]
    with pytest.raises(ValueError, match="1000"):
        checks_payload(checks)
