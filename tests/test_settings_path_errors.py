from pathlib import Path
from unittest.mock import patch

import pytest

from settings import env_path


def test_path_expansion_errors_are_normalized(monkeypatch):
    monkeypatch.setenv("WAKEUP_TEST_PATH", "~/state.db")
    for error in (OSError("home unavailable"), RuntimeError("home loop"), ValueError("bad path")):
        with patch.object(Path, "expanduser", side_effect=error), pytest.raises(ValueError, match="filesystem path"):
            env_path("WAKEUP_TEST_PATH", "state.db")
