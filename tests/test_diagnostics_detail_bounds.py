from diagnostics import Check, checks_payload


def test_diagnostic_details_are_bounded_for_machine_output():
    payload = checks_payload([Check("backend", False, "x" * 5000)])
    assert len(payload[0]["detail"]) == 1000
