"""The local classifier response is a small, strict yes/no boundary."""

import pytest

import perception


@pytest.mark.parametrize("value", ["yes", "Yes!", " yes. because it is recreational "])
def test_classifier_verdict_accepts_leading_yes_with_punctuation(value):
    assert perception._classifier_verdict(value) == "yes"


@pytest.mark.parametrize("value", ["no", "No!", " no: this is planned work "])
def test_classifier_verdict_accepts_leading_no_with_punctuation(value):
    assert perception._classifier_verdict(value) == "no"


@pytest.mark.parametrize("value", [None, 1, "", "maybe", "the answer is yes"])
def test_classifier_verdict_rejects_missing_or_nonleading_answers(value):
    assert perception._classifier_verdict(value) is None


def test_classifier_verdict_only_inspects_bounded_prefix():
    value = "unclear " + "x" * perception._MAX_CLASSIFIER_RESPONSE_CHARS + " yes"
    assert perception._classifier_verdict(value) is None
