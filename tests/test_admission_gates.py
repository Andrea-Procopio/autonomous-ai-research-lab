"""The admission gate: everything checkable is checked, nothing else.

Pure tests over parsed payloads plus trusted context. Every reject-list
item from the directive has a named firing test here, and the honest
payload passes untouched — the gate never raises on taste.
"""

from __future__ import annotations

from collections.abc import Mapping

from autonomous_research_lab.admission import check_operationalization
from autonomous_research_lab.runtime.providers import OutputSchema

_FIELD_TEXTS: Mapping[str, str] = {
    "hypothesis": "reweighted heads carry the in-context ability",
    "predictions[0].text": (
        "Ablating top-weighted heads drops accuracy substantially more "
        "than ablating bottom-weighted heads."
    ),
    "predictions[0].falsifier": (
        "If ablating top-weighted heads causes a similar drop to "
        "ablating bottom-weighted heads, the prediction fails."
    ),
    "metrics[0]": "few-shot accuracy",
    "resources.compute": "about one hundred GPU-hours on a single A100",
    "direction.scope": "mechanistic accounts of in-context learning",
}

_CANDIDATE_BLOCK = " \n ".join(
    _FIELD_TEXTS[key]
    for key in sorted(_FIELD_TEXTS)
    if not key.startswith("direction.")
)

_FALSIFIERS: Mapping[str, str] = {
    " ".join(_FIELD_TEXTS["predictions[0].text"].casefold().split()): (
        _FIELD_TEXTS["predictions[0].falsifier"]
    ),
}

_METRICS = ("few-shot accuracy", "prefix matching score")

_KNOWN_IDS = frozenset({"idea_1e1fa63952cc0d91", "dir_7229971404b1e968"})

_HAYSTACK = frozenset({"100"})


def _support(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "source": "candidate",
        "field_path": "predictions[0].text",
        "quote": "drops accuracy substantially more",
    }
    entry.update(overrides)
    return entry


def _encoding(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "prediction_text": _FIELD_TEXTS["predictions[0].text"],
        "condition": "few-shot classification with trained scalars",
        "base_metric": "few-shot accuracy",
        "expected_higher_arm": "ablating top-weighted heads",
        "expected_lower_arm": "ablating bottom-weighted heads",
        "contrary_observation": (
            "a similar drop to ablating bottom-weighted heads"
        ),
        "support": [_support()],
    }
    entry.update(overrides)
    return entry


def _payload(*encodings: Mapping[str, object]) -> dict[str, object]:
    return {"operational_predictions": list(encodings or (_encoding(),))}


def _check(payload: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (rejection.rule, rejection.detail)
        for rejection in check_operationalization(
            payload,
            field_texts=_FIELD_TEXTS,
            candidate_block=_CANDIDATE_BLOCK,
            falsifier_by_prediction=_FALSIFIERS,
            metrics=_METRICS,
            haystack_tokens=_HAYSTACK,
            known_ids=_KNOWN_IDS,
        )
    )


def _rules(payload: Mapping[str, object]) -> set[str]:
    return {rule for rule, _ in _check(payload)}


def test_a_grounded_payload_passes() -> None:
    assert _check(_payload()) == ()


def test_the_gate_never_raises_on_taste() -> None:
    """A cautious, hedged, even weak encoding is not a rule violation:
    the gate returns rules, never opinions, and an odd-but-grounded
    payload passes."""
    timid = _encoding(
        condition="a narrow setting where the effect might be smallest",
    )
    assert _check(_payload(timid)) == ()


def test_a_fabricated_reference_is_rejected() -> None:
    smuggled = _encoding(
        condition="as idea_deadbeefcafebabe already establishes"
    )
    assert "fabricated_reference" in _rules(_payload(smuggled))


def test_a_known_reference_is_not_fabricated() -> None:
    grounded = _encoding(
        condition="the setting idea_1e1fa63952cc0d91 records"
    )
    assert "fabricated_reference" not in _rules(_payload(grounded))


def test_an_unknown_field_path_is_rejected() -> None:
    bad = _encoding(support=[_support(field_path="reviews[0].verdict")])
    assert "unknown_field" in _rules(_payload(bad))


def test_a_mislabeled_source_is_rejected() -> None:
    mislabeled = _encoding(
        support=[_support(source="direction")]
    )
    assert "unknown_field" in _rules(_payload(mislabeled))


def test_a_paraphrased_quote_is_not_verbatim() -> None:
    paraphrase = _encoding(
        support=[_support(quote="accuracy declines quite a lot more")]
    )
    assert "unsupported_claim" in _rules(_payload(paraphrase))


def test_normalization_forgives_case_and_whitespace_only() -> None:
    renormalized = _encoding(
        support=[_support(quote="  Drops   ACCURACY substantially more ")]
    )
    assert _check(_payload(renormalized)) == ()


def test_a_rewritten_prediction_is_rejected() -> None:
    rewritten = _encoding(
        prediction_text="Top heads matter more than bottom heads."
    )
    rules = _rules(_payload(rewritten))
    assert "missing_support" in rules
    assert "missing_decision" in rules  # the recorded prediction is uncovered


def test_an_undeclared_metric_is_rejected() -> None:
    invented = _encoding(base_metric="overall goodness")
    assert "metric_not_declared" in _rules(_payload(invented))


def test_an_arm_the_record_does_not_contain_is_rejected() -> None:
    invented = _encoding(
        expected_higher_arm="pruning every attention layer"
    )
    assert "missing_support" in _rules(_payload(invented))


def test_a_self_comparison_is_circular() -> None:
    circular = _encoding(
        expected_lower_arm="Ablating  TOP-WEIGHTED heads"
    )
    assert "circular_finding" in _rules(_payload(circular))


def test_an_overlong_arm_is_malformed() -> None:
    padded = _encoding(
        expected_higher_arm="ablating top-weighted heads " + "x" * 120
    )
    assert "malformed_finding" in _rules(_payload(padded))


def test_a_contrary_quoted_from_the_prediction_itself_is_rejected() -> None:
    """The contrary outcome must come from the falsifier — quoting the
    prediction's own text as its contrary would let an encoding refute
    nothing."""
    circular = _encoding(
        contrary_observation="drops accuracy substantially more"
    )
    assert "missing_support" in _rules(_payload(circular))


def test_a_skipped_prediction_is_rejected() -> None:
    assert "missing_decision" in _rules(
        {"operational_predictions": []}
    )


def test_too_many_encodings_violate_the_budget() -> None:
    conditions = (
        "condition one",
        "condition two",
        "condition three",
        "condition four",
    )
    encodings = tuple(
        _encoding(condition=condition) for condition in conditions
    )
    assert "budget_violation" in _rules(_payload(*encodings))


def test_a_repeated_mechanical_tuple_is_a_duplicate() -> None:
    """Two encodings with the same condition, metric, and arms would
    derive the same core prediction id (its content id excludes the
    prose expectation), so the duplicate is caught here, on the
    mechanical tuple, never on the prose."""
    first = _encoding()
    second = _encoding(
        contrary_observation="ablating bottom-weighted heads"
    )
    assert "duplicate_finding" in _rules(_payload(first, second))


def test_a_repeated_support_link_is_a_duplicate() -> None:
    doubled = _encoding(support=[_support(), _support()])
    assert "duplicate_finding" in _rules(_payload(doubled))


def test_missing_support_links_are_rejected() -> None:
    bare = _encoding(support=[])
    assert "missing_support" in _rules(_payload(bare))


def test_novelty_language_is_rejected() -> None:
    grand = _encoding(condition="a novel setting nobody has explored")
    assert "novelty_claim" in _rules(_payload(grand))


def test_guarantee_language_is_rejected() -> None:
    certain = _encoding(condition="this experiment cannot fail")
    assert "novelty_claim" in _rules(_payload(certain))


def test_honest_uncertainty_does_not_fire_the_guarantee_rule() -> None:
    """The guarantee pattern is word-bounded: "uncertainty" prose must
    never trip it."""
    hedged = _encoding(
        condition="a setting with real uncertainty about the mechanism"
    )
    assert "novelty_claim" not in _rules(_payload(hedged))


def test_an_ungrounded_number_is_rejected() -> None:
    numeric = _encoding(condition="an effect of at least 0.35 accuracy")
    assert "ungrounded_number" in _rules(_payload(numeric))


def test_a_grounded_number_passes() -> None:
    grounded = _encoding(condition="within the 100 GPU-hour envelope")
    assert "ungrounded_number" not in _rules(_payload(grounded))


def test_empty_prose_is_rejected() -> None:
    hollow = _encoding(condition="   ")
    assert "empty_finding" in _rules(_payload(hollow))


def test_malformed_entries_are_named() -> None:
    assert "malformed_finding" in _rules(
        {"operational_predictions": ["not an object"]}
    )


def test_a_schema_frozen_payload_reads_the_same_as_a_plain_one() -> None:
    """OutputSchema.parse deep-freezes arrays into tuples; the gate must
    read a frozen payload exactly like a hand-built one (the 5E
    regression)."""
    schema = OutputSchema(
        name="operationalization",
        json_schema={
            "type": "object",
            "properties": {
                "operational_predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prediction_text": {"type": "string"},
                            "condition": {"type": "string"},
                            "base_metric": {"type": "string"},
                            "expected_higher_arm": {"type": "string"},
                            "expected_lower_arm": {"type": "string"},
                            "contrary_observation": {"type": "string"},
                            "support": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "field_path": {"type": "string"},
                                        "quote": {"type": "string"},
                                    },
                                    "required": [
                                        "source",
                                        "field_path",
                                        "quote",
                                    ],
                                },
                            },
                        },
                        "required": [
                            "prediction_text",
                            "condition",
                            "base_metric",
                            "expected_higher_arm",
                            "expected_lower_arm",
                            "contrary_observation",
                            "support",
                        ],
                    },
                }
            },
            "required": ["operational_predictions"],
        },
    )
    import json

    frozen = schema.parse(json.dumps(_payload()))
    plain_result = _check(_payload())
    frozen_result = tuple(
        (rejection.rule, rejection.detail)
        for rejection in check_operationalization(
            frozen,
            field_texts=_FIELD_TEXTS,
            candidate_block=_CANDIDATE_BLOCK,
            falsifier_by_prediction=_FALSIFIERS,
            metrics=_METRICS,
            haystack_tokens=_HAYSTACK,
            known_ids=_KNOWN_IDS,
        )
    )
    assert frozen_result == plain_result == ()


def test_every_fired_rule_is_returned_together() -> None:
    """Collect-all, never first-fail: one badly wrong payload names all
    of its problems at once."""
    wreck = _encoding(
        base_metric="overall goodness",
        expected_higher_arm="pruning every attention layer",
        condition="a novel effect of 0.35 via idea_deadbeefcafebabe",
        support=[_support(quote="accuracy declines quite a lot more")],
    )
    rules = _rules(_payload(wreck))
    assert {
        "metric_not_declared",
        "missing_support",
        "novelty_claim",
        "ungrounded_number",
        "fabricated_reference",
        "unsupported_claim",
    } <= rules
