from decimal import Decimal

import pytest

from history import ContextHistory


@pytest.mark.parametrize("version", [True, 1.0, Decimal("1"), "1"])
def test_restore_rejects_non_integer_version_values(version):
    snapshot = ContextHistory().snapshot()
    snapshot["version"] = version

    with pytest.raises(ValueError, match="version 1"):
        ContextHistory.from_snapshot(snapshot)
