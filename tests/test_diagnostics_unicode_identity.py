import pytest

from diagnostics import Check, format_checks


def test_diagnostics_reject_unicode_equivalent_check_names():
    checks = [Check("python", True, "ok"), Check("ｐｙｔｈｏｎ", True, "also ok")]

    with pytest.raises(ValueError, match="duplicate diagnostic check name"):
        format_checks(checks)
