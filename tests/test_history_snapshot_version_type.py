import pytest

from history import ContextHistory


def test_restore_rejects_boolean_version_values():
    snapshot = ContextHistory().snapshot()
    snapshot["version"] = True

    with pytest.raises(ValueError, match="version 1"):
        ContextHistory.from_snapshot(snapshot)
