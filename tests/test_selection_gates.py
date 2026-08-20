"""The selection gates: complete feedback, verbatim attestation at both
ends, exact restatement of recorded verdicts, and no route to a stop
while a defensible candidate remains. Never a rule about taste."""

from __future__ import annotations

from collections.abc import Mapping

from autonomous_research_lab.selection.gates import (
    check_comparative_review,
    check_selection_decision,
)
from autonomous_research_lab.selection.records import REVIEW_FIELDS

BLOCK_A = (
    "## Candidate idea_a\n"
    "title: Head Reweighting\n"
    "hypothesis: reweighted heads carry the ability\n"
    "resources: compute: one GPU-day; data: synthetic; implementation: small\n"
    "risks: effects may not localize\n"
    "metrics: accuracy on 3 probe tasks\n"
)
BLOCK_B = (
    "## Candidate idea_b\n"
    "title: Curriculum Distillation\n"
    "hypothesis: staged distillation preserves rare skills\n"
    "resources: compute: a 64-GPU cluster for a week; data: public corpora\n"
    "risks: the effect may wash out at scale\n"
)

BLOCKS = {"idea_a": BLOCK_A, "idea_b": BLOCK_B}
VERDICTS = {"idea_a": "distinguished", "idea_b": "distinguished"}
CONSTRAINTS = {
    "compute": "One CPU workstation; no GPU cluster is available.",
    "data": "Public datasets only.",
    "time": "Runs finish within hours.",
    "experimental": "Containerized seeded runs.",
    "scope": "mechanistic accounts of in-context learning",
}
TOKENS = frozenset({"3", "64"})
KNOWN_IDS = frozenset({"idea_a", "idea_b"})


def _review_entry(candidate_id: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "candidate_id": candidate_id,
        "prior_art_verdict": "distinguished",
        "disqualifiers": [],
    }
    for name in REVIEW_FIELDS:
        entry[name] = f"{name} weighed in plain prose"
    entry.update(overrides)
    return entry


def _disqualifier_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "ground": "resources_exceed_directive",
        "dimension": "compute",
        "candidate_text": "a 64-GPU cluster for a week",
        "constraint_text": "no GPU cluster is available",
        "why_unrepairable": "shrinking the cluster changes the hypothesis",
    }
    entry.update(overrides)
    return entry


def _pair_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "first_candidate_id": "idea_a",
        "second_candidate_id": "idea_b",
        "comparison": "idea_a is cheaper to falsify",
    }
    entry.update(overrides)
    return entry


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "reviews": [_review_entry("idea_a"), _review_entry("idea_b")],
        "pairwise_comparisons": [_pair_entry()],
    }
    payload.update(overrides)
    return payload


def _review_rules(payload: Mapping[str, object]) -> set[str]:
    return {
        rejection.rule
        for rejection in check_comparative_review(
            payload,
            candidate_blocks=BLOCKS,
            assessment_verdicts=VERDICTS,
            constraint_haystacks=CONSTRAINTS,
            haystack_tokens=TOKENS,
            known_ids=KNOWN_IDS,
        )
    }


def _decision_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "selected_candidate_id": "idea_a",
        "decisive_tradeoff": "a sharper falsifier at equal cost",
        "why_selected_over": [
            {"candidate_id": "idea_b", "reason": "weaker falsifier"}
        ],
        "first_experimental_objective": "reproduce the probe baseline",
        "required_capabilities": ["dataset download"],
        "residual_risks": ["the effect may be dataset-specific"],
    }
    payload.update(overrides)
    return payload


def _decision_rules(
    payload: Mapping[str, object],
    disqualified: frozenset[str] = frozenset(),
) -> set[str]:
    return {
        rejection.rule
        for rejection in check_selection_decision(
            payload,
            eligible_ids=frozenset(BLOCKS),
            disqualified_ids=disqualified,
            haystack_tokens=TOKENS,
            known_ids=KNOWN_IDS,
        )
    }


# -- stage 1: the comparative review ---------------------------------------------


def test_a_grounded_comparative_review_passes() -> None:
    assert _review_rules(_payload()) == set()


def test_every_eligible_pair_is_compared_exactly_once() -> None:
    assert "missing_decision" in _review_rules(
        _payload(pairwise_comparisons=[])
    )
    assert "duplicate_finding" in _review_rules(
        _payload(
            pairwise_comparisons=[
                _pair_entry(),
                _pair_entry(comparison="said twice"),
            ]
        )
    )
    assert "budget_violation" in _review_rules(
        _payload(
            pairwise_comparisons=[
                _pair_entry(),
                _pair_entry(comparison="said twice"),
            ]
        )
    )


def test_a_self_pair_is_circular() -> None:
    assert "circular_finding" in _review_rules(
        _payload(
            pairwise_comparisons=[_pair_entry(second_candidate_id="idea_a")]
        )
    )


def test_a_review_over_an_unknown_candidate_is_rejected() -> None:
    rules = _review_rules(
        _payload(
            reviews=[
                _review_entry("idea_a"),
                _review_entry("idea_b"),
                _review_entry("idea_zz"),
            ]
        )
    )
    assert "unknown_candidate" in rules


def test_a_review_cannot_shrink_or_grow_the_eligible_set() -> None:
    # Dropping an eligible candidate is a missing decision; inventing one
    # is an unknown candidate. Either way the stamped set stands.
    assert "missing_decision" in _review_rules(
        _payload(
            reviews=[_review_entry("idea_a")], pairwise_comparisons=[]
        )
    )
    assert "unknown_candidate" in _review_rules(
        _payload(
            reviews=[
                _review_entry("idea_a"),
                _review_entry("idea_zz"),
            ],
            pairwise_comparisons=[_pair_entry()],
        )
    )


def test_a_duplicate_review_is_rejected() -> None:
    assert "duplicate_finding" in _review_rules(
        _payload(
            reviews=[
                _review_entry("idea_a"),
                _review_entry("idea_a"),
                _review_entry("idea_b"),
            ]
        )
    )


def test_a_misstated_verdict_is_rejected() -> None:
    rules = _review_rules(
        _payload(
            reviews=[
                _review_entry(
                    "idea_a", prior_art_verdict="novelty_unresolved"
                ),
                _review_entry("idea_b"),
            ]
        )
    )
    assert "misstated_verdict" in rules


def test_an_honest_verdict_restatement_is_not_a_misstatement() -> None:
    # The gate compares to the record; it never bans verdict words.
    entry = _review_entry(
        "idea_a",
        prior_art_differentiation=(
            "distinguished within the challenged corpus from four "
            "compared works"
        ),
    )
    rules = _review_rules(
        _payload(reviews=[entry, _review_entry("idea_b")])
    )
    assert "misstated_verdict" not in rules
    assert "novelty_claim" not in rules


def test_a_rationale_may_not_claim_novelty() -> None:
    entry = _review_entry(
        "idea_a", scientific_importance="a novel and unexplored direction"
    )
    assert "novelty_claim" in _review_rules(
        _payload(reviews=[entry, _review_entry("idea_b")])
    )


def test_coverage_language_is_rejected() -> None:
    entry = _review_entry(
        "idea_a", prior_art_differentiation="no prior work exists on this"
    )
    assert "coverage_language" in _review_rules(
        _payload(reviews=[entry, _review_entry("idea_b")])
    )


def test_ungrounded_numbers_are_rejected() -> None:
    entry = _review_entry(
        "idea_a", expected_information_gain="a 37 percent gain is likely"
    )
    assert "ungrounded_number" in _review_rules(
        _payload(reviews=[entry, _review_entry("idea_b")])
    )


def test_numbers_from_the_rendered_records_are_grounded() -> None:
    entry = _review_entry(
        "idea_a", evaluation_quality="3 probe tasks give a clean readout"
    )
    assert "ungrounded_number" not in _review_rules(
        _payload(reviews=[entry, _review_entry("idea_b")])
    )


# -- disqualifier attestation ------------------------------------------------------


def test_an_attested_disqualifier_passes() -> None:
    entry = _review_entry(
        "idea_b", disqualifiers=[_disqualifier_entry()]
    )
    assert _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    ) == set()


def test_an_unattested_candidate_quote_is_rejected() -> None:
    entry = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(
                candidate_text="a paraphrase of the cluster need"
            )
        ],
    )
    assert "missing_support" in _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    )


def test_an_unattested_constraint_quote_is_rejected() -> None:
    entry = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(constraint_text="no clusters, ever")
        ],
    )
    assert "unsupported_claim" in _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    )


def test_a_paraphrased_quote_is_not_verbatim() -> None:
    # Case and whitespace are forgiven; wording is not.
    forgiven = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(
                candidate_text="A   64-GPU cluster\nfor a week"
            )
        ],
    )
    assert _review_rules(
        _payload(reviews=[_review_entry("idea_a"), forgiven])
    ) == set()
    reworded = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(candidate_text="a sixty-four GPU cluster")
        ],
    )
    assert "missing_support" in _review_rules(
        _payload(reviews=[_review_entry("idea_a"), reworded])
    )


def test_a_mismatched_ground_and_dimension_contradicts() -> None:
    entry = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(
                ground="outside_cfp_scope", dimension="compute"
            )
        ],
    )
    assert "disqualifier_contradiction" in _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    )


def test_a_scope_disqualifier_quotes_the_direction() -> None:
    entry = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(
                ground="outside_cfp_scope",
                dimension="scope",
                constraint_text=(
                    "mechanistic accounts of in-context learning"
                ),
            )
        ],
    )
    assert _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    ) == set()


def test_a_repeated_ground_is_a_duplicate() -> None:
    entry = _review_entry(
        "idea_b",
        disqualifiers=[
            _disqualifier_entry(),
            _disqualifier_entry(why_unrepairable="said again"),
        ],
    )
    assert "duplicate_finding" in _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    )


def test_all_fired_rules_are_returned_together() -> None:
    entry = _review_entry(
        "idea_b",
        prior_art_verdict="overlapping",
        scientific_importance="a novel gain of 37 percent",
        disqualifiers=[
            _disqualifier_entry(candidate_text="a paraphrase"),
        ],
    )
    rules = _review_rules(
        _payload(reviews=[_review_entry("idea_a"), entry])
    )
    assert {
        "misstated_verdict",
        "novelty_claim",
        "ungrounded_number",
        "missing_support",
    } <= rules


# -- stage 2: the decision ----------------------------------------------------------


def test_a_grounded_decision_passes() -> None:
    assert _decision_rules(_decision_payload()) == set()


def test_the_winner_must_be_eligible() -> None:
    assert "unknown_candidate" in _decision_rules(
        _decision_payload(selected_candidate_id="idea_zz")
    )


def test_a_disqualified_winner_has_no_route_to_a_record() -> None:
    rules = _decision_rules(
        _decision_payload(
            selected_candidate_id="idea_b",
            why_selected_over=[
                {"candidate_id": "idea_a", "reason": "weaker"}
            ],
        ),
        disqualified=frozenset({"idea_b"}),
    )
    assert "disqualified_selection" in rules


def test_stopping_is_inexpressible_so_contenders_must_be_argued() -> None:
    # There is no stop shape: a payload that simply skips a contender is
    # an incomplete decision, not an honest refusal.
    assert "missing_decision" in _decision_rules(
        _decision_payload(why_selected_over=[])
    )


def test_the_winner_is_not_argued_against_itself() -> None:
    assert "circular_finding" in _decision_rules(
        _decision_payload(
            why_selected_over=[
                {"candidate_id": "idea_a", "reason": "beats itself"},
                {"candidate_id": "idea_b", "reason": "weaker"},
            ]
        )
    )


def test_an_already_settled_candidate_is_not_argued_again() -> None:
    rules = _decision_rules(
        _decision_payload(
            why_selected_over=[
                {"candidate_id": "idea_b", "reason": "already settled"}
            ]
        ),
        disqualified=frozenset({"idea_b"}),
    )
    assert "disqualified_selection" in rules


def test_empty_capability_and_risk_lists_are_rejected() -> None:
    assert "empty_finding" in _decision_rules(
        _decision_payload(required_capabilities=[])
    )
    assert "empty_finding" in _decision_rules(
        _decision_payload(residual_risks=[])
    )


def test_decision_prose_is_held_to_the_text_discipline() -> None:
    assert "novelty_claim" in _decision_rules(
        _decision_payload(decisive_tradeoff="the novel option wins")
    )
    assert "ungrounded_number" in _decision_rules(
        _decision_payload(
            first_experimental_objective="hit 99 percent accuracy"
        )
    )


def test_a_rationale_may_describe_a_candidate_in_its_own_words() -> None:
    rules = _decision_rules(
        _decision_payload(
            why_selected_over=[
                {
                    "candidate_id": "idea_b",
                    "reason": (
                        "its 64-GPU appetite crowds out replication; "
                        "idea_a answers its question with cheap probes"
                    ),
                }
            ]
        )
    )
    assert rules == set()


def test_a_schema_frozen_payload_reads_the_same_as_a_plain_one() -> None:
    """OutputSchema.parse deep-freezes arrays into tuples; the gate must
    read the frozen shape identically to a hand-built one — a frozen
    array silently reading as empty would fail every review as
    missing."""
    import json

    from autonomous_research_lab.selection.selector import (
        COMPARATIVE_REVIEW_SCHEMA,
    )

    plain = _payload()
    frozen = COMPARATIVE_REVIEW_SCHEMA.parse(json.dumps(plain))
    assert _review_rules(frozen) == _review_rules(plain) == set()


def test_the_gate_never_raises_on_taste() -> None:
    # Weird-but-lawful prose passes both gates: no rule is about style.
    entry = _review_entry(
        "idea_a",
        portfolio_redundancy=(
            "BOTH candidates orbit attention; so be it — orbits differ"
        ),
    )
    assert _review_rules(
        _payload(reviews=[entry, _review_entry("idea_b")])
    ) == set()
    assert _decision_rules(
        _decision_payload(
            decisive_tradeoff="pick the cheap falsifier; sleep at night"
        )
    ) == set()
