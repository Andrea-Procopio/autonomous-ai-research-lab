"""The selection vocabulary: one named run, an exact partition, three
structurally distinct outcomes, and score-free records throughout."""

from __future__ import annotations

import dataclasses

import pytest

from autonomous_research_lab.mapping.records import CallProvenance
from autonomous_research_lab.priorart.assessment import (
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtVerdict,
)
from autonomous_research_lab.selection.directive import (
    ELIGIBLE_CANDIDATES_CEILING,
    MAX_CONSTRAINT_CHARS,
    MODEL_CALLS_CEILING,
    SelectionDirective,
)
from autonomous_research_lab.selection.records import (
    CLAIM_KINDS,
    GROUND_DIMENSIONS,
    REVIEW_FIELDS,
    CandidateReview,
    DisqualificationGround,
    DisqualifierDimension,
    HardDisqualifier,
    IneligibleCandidate,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRationale,
    SelectionRunRecord,
)

RUN = "sel_1"


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id="req-9",
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=50,
        repair_count=0,
    )


def _directive(**overrides: object) -> SelectionDirective:
    values: dict[str, object] = {
        "prior_art_run_record_id": "prun_1",
        "compute_constraint": "One CPU workstation.",
        "data_constraint": "Public datasets only.",
        "time_constraint": "Runs finish within hours.",
        "experimental_constraint": "Containerized seeded runs.",
    }
    values.update(overrides)
    return SelectionDirective(**values)  # type: ignore[arg-type]


def _disqualifier(**overrides: object) -> HardDisqualifier:
    values: dict[str, object] = {
        "ground": DisqualificationGround.RESOURCES_EXCEED_DIRECTIVE,
        "dimension": DisqualifierDimension.COMPUTE,
        "candidate_text": "requires a 64-GPU cluster",
        "constraint_text": "One CPU workstation.",
        "why_unrepairable": "shrinking the cluster changes the hypothesis",
    }
    values.update(overrides)
    return HardDisqualifier(**values)  # type: ignore[arg-type]


def _review(
    candidate_id: str,
    disqualifiers: tuple[HardDisqualifier, ...] = (),
) -> CandidateReview:
    prose = {name: f"{name} judged for {candidate_id}" for name in REVIEW_FIELDS}
    return CandidateReview(
        candidate_id=candidate_id,
        prior_art_verdict=PriorArtVerdict.DISTINGUISHED,
        disqualifiers=disqualifiers,
        **prose,
    )


def _pair(first: str, second: str) -> PairwiseComparison:
    return PairwiseComparison(
        first_candidate_id=first,
        second_candidate_id=second,
        comparison=f"{first} is narrower than {second}",
    )


def _ineligible(
    candidate_id: str,
    verdict: PriorArtVerdict = PriorArtVerdict.NOVELTY_UNRESOLVED,
) -> IneligibleCandidate:
    if verdict is PriorArtVerdict.OVERLAPPING:
        return IneligibleCandidate(
            candidate_id=candidate_id,
            assessment_id="paa_o",
            verdict=verdict,
            reasons=(),
            overlapping_work_ids=("lit_1",),
        )
    return IneligibleCandidate(
        candidate_id=candidate_id,
        assessment_id="paa_u",
        verdict=verdict,
        reasons=(
            PriorArtReason(
                PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES, "thin pool"
            ),
        ),
        overlapping_work_ids=(),
    )


def _decision(winner: str, others: tuple[str, ...]) -> SelectionDecision:
    return SelectionDecision(
        selected_candidate_id=winner,
        decisive_tradeoff="sharper falsifier under the same budget",
        why_selected_over=tuple(
            SelectionRationale(candidate_id=other, reason=f"beats {other}")
            for other in others
        ),
        first_experimental_objective="reproduce the baseline",
        required_capabilities=("dataset download",),
        residual_risks=("effect may be dataset-specific",),
        provenance=_provenance(),
    )


def _selected_run(**overrides: object) -> SelectionRunRecord:
    values: dict[str, object] = {
        "run_id": RUN,
        "directive_id": "sdir_1",
        "prior_art_run_record_id": "prun_1",
        "prior_art_run_id": "pac_1",
        "ideation_run_record_id": "irun_1",
        "ideation_run_id": "idg_1",
        "direction_id": "dir_1",
        "candidate_ids": ("idea_a", "idea_b", "idea_c"),
        "prior_art_assessment_ids": ("paa_a", "paa_b", "paa_c"),
        "eligible_candidate_ids": ("idea_a", "idea_b"),
        "ineligible": (_ineligible("idea_c"),),
        "disqualified_candidate_ids": (),
        "reviews": (_review("idea_a"), _review("idea_b")),
        "pairwise_comparisons": (_pair("idea_a", "idea_b"),),
        "review_provenance": _provenance(),
        "outcome": SelectionOutcome.SELECTED,
        "decision": _decision("idea_a", ("idea_b",)),
        "model_calls": 2,
        "input_tokens": 900,
        "output_tokens": 400,
    }
    values.update(overrides)
    return SelectionRunRecord(**values)  # type: ignore[arg-type]


def _no_eligible_run(**overrides: object) -> SelectionRunRecord:
    values: dict[str, object] = {
        "run_id": RUN,
        "directive_id": "sdir_1",
        "prior_art_run_record_id": "prun_1",
        "prior_art_run_id": "pac_1",
        "ideation_run_record_id": "irun_1",
        "ideation_run_id": "idg_1",
        "direction_id": "dir_1",
        "candidate_ids": ("idea_a", "idea_b"),
        "prior_art_assessment_ids": ("paa_a", "paa_b"),
        "eligible_candidate_ids": (),
        "ineligible": (
            _ineligible("idea_a", PriorArtVerdict.OVERLAPPING),
            _ineligible("idea_b"),
        ),
        "disqualified_candidate_ids": (),
        "reviews": (),
        "pairwise_comparisons": (),
        "review_provenance": None,
        "outcome": SelectionOutcome.NO_ELIGIBLE_CANDIDATE,
        "decision": None,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    values.update(overrides)
    return SelectionRunRecord(**values)  # type: ignore[arg-type]


def _no_defensible_run(**overrides: object) -> SelectionRunRecord:
    values: dict[str, object] = {
        "run_id": RUN,
        "directive_id": "sdir_1",
        "prior_art_run_record_id": "prun_1",
        "prior_art_run_id": "pac_1",
        "ideation_run_record_id": "irun_1",
        "ideation_run_id": "idg_1",
        "direction_id": "dir_1",
        "candidate_ids": ("idea_a", "idea_b"),
        "prior_art_assessment_ids": ("paa_a", "paa_b"),
        "eligible_candidate_ids": ("idea_a", "idea_b"),
        "ineligible": (),
        "disqualified_candidate_ids": ("idea_a", "idea_b"),
        "reviews": (
            _review("idea_a", (_disqualifier(),)),
            _review("idea_b", (_disqualifier(),)),
        ),
        "pairwise_comparisons": (_pair("idea_a", "idea_b"),),
        "review_provenance": _provenance(),
        "outcome": SelectionOutcome.NO_DEFENSIBLE_CANDIDATE,
        "decision": None,
        "model_calls": 1,
        "input_tokens": 500,
        "output_tokens": 200,
    }
    values.update(overrides)
    return SelectionRunRecord(**values)  # type: ignore[arg-type]


# -- the directive -------------------------------------------------------------


def test_a_directive_names_exactly_one_prior_art_run() -> None:
    with pytest.raises(ValueError, match="prior-art run record"):
        _directive(prior_art_run_record_id="  ")


def test_directive_identity_is_deterministic() -> None:
    assert _directive().id == _directive().id
    assert _directive().id != _directive(time_constraint="Weeks.").id
    assert _directive().id.startswith("sdir_")


def test_ceiling_defaults_are_coherent() -> None:
    directive = _directive(
        max_eligible_candidates=ELIGIBLE_CANDIDATES_CEILING,
        max_model_calls=MODEL_CALLS_CEILING,
    )
    assert directive.max_eligible_candidates == ELIGIBLE_CANDIDATES_CEILING
    assert _directive().max_model_calls == MODEL_CALLS_CEILING


def test_an_incoherent_ceiling_is_unconstructible() -> None:
    with pytest.raises(ValueError, match="max_eligible_candidates"):
        _directive(max_eligible_candidates=0)
    with pytest.raises(ValueError, match="max_eligible_candidates"):
        _directive(max_eligible_candidates=ELIGIBLE_CANDIDATES_CEILING + 1)
    with pytest.raises(ValueError, match="max_model_calls"):
        _directive(max_model_calls=MODEL_CALLS_CEILING + 1)


def test_a_constraint_must_be_a_short_statement() -> None:
    with pytest.raises(ValueError, match="compute_constraint"):
        _directive(compute_constraint="")
    with pytest.raises(ValueError, match="short statement"):
        _directive(data_constraint="x" * (MAX_CONSTRAINT_CHARS + 1))


# -- the closed universes -------------------------------------------------------


def test_the_outcome_universe_is_closed() -> None:
    assert {outcome.value for outcome in SelectionOutcome} == {
        "selected",
        "no_eligible_candidate",
        "no_defensible_candidate",
    }


def test_the_claim_kind_universe_is_closed() -> None:
    assert set(CLAIM_KINDS.values()) == {
        "record_restatement",
        "candidate_grounded_judgment",
        "record_quotation",
        "comparative_preference",
        "design_target",
    }
    for name in REVIEW_FIELDS:
        assert CLAIM_KINDS[f"review.{name}"] == "candidate_grounded_judgment"
    assert (
        CLAIM_KINDS["decision.selected_candidate_id"]
        == "comparative_preference"
    )
    assert CLAIM_KINDS["review.disqualifier.candidate_text"] == (
        "record_quotation"
    )


def test_every_ground_names_its_quotable_dimensions() -> None:
    assert set(GROUND_DIMENSIONS) == set(DisqualificationGround)
    assert GROUND_DIMENSIONS[DisqualificationGround.OUTSIDE_CFP_SCOPE] == {
        DisqualifierDimension.SCOPE
    }


# -- disqualifiers ---------------------------------------------------------------


def test_a_disqualifier_requires_both_of_its_texts() -> None:
    with pytest.raises(ValueError, match="candidate_text"):
        _disqualifier(candidate_text=" ")
    with pytest.raises(ValueError, match="constraint_text"):
        _disqualifier(constraint_text=" ")
    with pytest.raises(ValueError, match="why_unrepairable"):
        _disqualifier(why_unrepairable=" ")


def test_a_mismatched_ground_and_dimension_is_unconstructible() -> None:
    with pytest.raises(ValueError, match="cannot quote"):
        _disqualifier(
            ground=DisqualificationGround.OUTSIDE_CFP_SCOPE,
            dimension=DisqualifierDimension.COMPUTE,
        )


def test_a_review_carries_at_most_one_disqualifier_per_ground() -> None:
    with pytest.raises(ValueError, match="one disqualifier per ground"):
        _review("idea_a", (_disqualifier(), _disqualifier()))


# -- pairs and copies ------------------------------------------------------------


def test_pairs_are_canonical_and_never_reflexive() -> None:
    with pytest.raises(ValueError, match="distinct"):
        _pair("idea_a", "idea_a")
    with pytest.raises(ValueError, match="canonical"):
        _pair("idea_b", "idea_a")


def test_an_ineligible_copy_reenforces_the_assessment_invariants() -> None:
    with pytest.raises(ValueError, match="eligible"):
        _ineligible("idea_a", PriorArtVerdict.DISTINGUISHED)
    with pytest.raises(ValueError, match="imply each other"):
        IneligibleCandidate(
            candidate_id="idea_a",
            assessment_id="paa_a",
            verdict=PriorArtVerdict.OVERLAPPING,
            reasons=(),
            overlapping_work_ids=(),
        )
    with pytest.raises(ValueError, match="names why"):
        IneligibleCandidate(
            candidate_id="idea_a",
            assessment_id="paa_a",
            verdict=PriorArtVerdict.NOVELTY_UNRESOLVED,
            reasons=(),
            overlapping_work_ids=(),
        )


# -- the three outcome shapes ------------------------------------------------------


def test_a_selected_record_names_its_survivor_and_the_rest() -> None:
    record = _selected_run()
    assert record.outcome is SelectionOutcome.SELECTED
    assert record.decision is not None
    assert record.decision.selected_candidate_id == "idea_a"
    assert {
        entry.candidate_id for entry in record.decision.why_selected_over
    } == {"idea_b"}
    assert record.ineligible[0].candidate_id == "idea_c"
    assert record.id.startswith("srun_")


def test_a_selected_winner_must_be_an_undisqualified_contender() -> None:
    with pytest.raises(ValueError, match="no validated"):
        _selected_run(
            disqualified_candidate_ids=("idea_a",),
            reviews=(
                _review("idea_a", (_disqualifier(),)),
                _review("idea_b"),
            ),
            decision=_decision("idea_a", ("idea_b",)),
        )
    with pytest.raises(ValueError, match="no validated"):
        _selected_run(decision=_decision("idea_zz", ("idea_b",)))


def test_a_selected_decision_argues_against_every_contender() -> None:
    with pytest.raises(ValueError, match="exactly the other"):
        _selected_run(decision=_decision("idea_a", ()))


def test_a_no_eligible_record_names_every_ineligible_candidate() -> None:
    record = _no_eligible_run()
    assert record.outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE
    verdicts = {
        entry.candidate_id: entry.verdict for entry in record.ineligible
    }
    assert verdicts == {
        "idea_a": PriorArtVerdict.OVERLAPPING,
        "idea_b": PriorArtVerdict.NOVELTY_UNRESOLVED,
    }
    assert record.ineligible[0].overlapping_work_ids == ("lit_1",)
    assert record.ineligible[1].reasons[0].code is (
        PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES
    )


def test_a_no_eligible_record_is_structurally_free() -> None:
    with pytest.raises(ValueError, match="cannot have spent"):
        _no_eligible_run(model_calls=1)
    with pytest.raises(ValueError, match="cannot have spent"):
        _no_eligible_run(output_tokens=10)
    with pytest.raises(ValueError, match="no model call"):
        _no_eligible_run(review_provenance=_provenance())


def test_a_no_defensible_record_requires_a_disqualifier_per_eligible() -> None:
    record = _no_defensible_run()
    assert record.outcome is SelectionOutcome.NO_DEFENSIBLE_CANDIDATE
    with pytest.raises(ValueError, match="defensible candidate remains"):
        _no_defensible_run(
            disqualified_candidate_ids=("idea_a",),
            reviews=(
                _review("idea_a", (_disqualifier(),)),
                _review("idea_b"),
            ),
        )


def test_the_disqualified_stamp_must_match_the_validated_evidence() -> None:
    with pytest.raises(ValueError, match="validated disqualifiers"):
        _selected_run(disqualified_candidate_ids=("idea_b",))


def test_the_partition_must_be_exact() -> None:
    with pytest.raises(ValueError, match="partition"):
        _selected_run(ineligible=())
    with pytest.raises(ValueError, match="partition"):
        _no_eligible_run(ineligible=(_ineligible("idea_a"),))


def test_reviews_and_pairs_cover_the_eligible_set_exactly() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        _selected_run(reviews=(_review("idea_a"),))
    with pytest.raises(ValueError, match="every eligible pair"):
        _selected_run(pairwise_comparisons=())


def test_record_identity_is_deterministic() -> None:
    assert _selected_run().id == _selected_run().id
    assert _no_eligible_run().id == _no_eligible_run().id
    assert _selected_run().id != _selected_run(input_tokens=901).id


# -- score-free by construction ------------------------------------------------------


def test_records_declare_no_score_fields() -> None:
    banned = ("score", "rank", "rating", "weight", "probability")
    for record_type in (
        SelectionDirective,
        HardDisqualifier,
        CandidateReview,
        PairwiseComparison,
        IneligibleCandidate,
        SelectionRationale,
        SelectionDecision,
        SelectionRunRecord,
    ):
        for entry in dataclasses.fields(record_type):
            lowered = entry.name.lower()
            assert not any(term in lowered for term in banned), (
                f"{record_type.__name__}.{entry.name} smells like a score"
            )
            assert "float" not in str(entry.type), (
                f"{record_type.__name__}.{entry.name} admits a number "
                f"where only prose belongs"
            )
