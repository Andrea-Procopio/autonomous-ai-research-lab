"""The selection store: write-once, verify-on-repeat, tamper-loud, with
one account of each run and every rejected payload preserved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.mapping.records import CallProvenance
from autonomous_research_lab.priorart.assessment import (
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtVerdict,
)
from autonomous_research_lab.selection.directive import SelectionDirective
from autonomous_research_lab.selection.records import (
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
from autonomous_research_lab.selection.store import (
    SelectionConflictError,
    SelectionIntegrityError,
    SelectionStore,
)


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=50,
        repair_count=1,
    )


def _directive() -> SelectionDirective:
    return SelectionDirective(
        prior_art_run_record_id="prun_1",
        compute_constraint="One CPU workstation.",
        data_constraint="Public datasets only.",
        time_constraint="Runs finish within hours.",
        experimental_constraint="Containerized seeded runs.",
    )


def _review(
    candidate_id: str,
    disqualifiers: tuple[HardDisqualifier, ...] = (),
) -> CandidateReview:
    prose = {name: f"{name} for {candidate_id}" for name in REVIEW_FIELDS}
    return CandidateReview(
        candidate_id=candidate_id,
        prior_art_verdict=PriorArtVerdict.DISTINGUISHED,
        disqualifiers=disqualifiers,
        **prose,
    )


def _selected_run(run_id: str = "sel_1", tokens: int = 400) -> SelectionRunRecord:
    return SelectionRunRecord(
        run_id=run_id,
        directive_id=_directive().id,
        prior_art_run_record_id="prun_1",
        prior_art_run_id="pac_1",
        ideation_run_record_id="irun_1",
        ideation_run_id="idg_1",
        direction_id="dir_1",
        candidate_ids=("idea_a", "idea_b", "idea_c", "idea_d"),
        prior_art_assessment_ids=("paa_a", "paa_b", "paa_c", "paa_d"),
        eligible_candidate_ids=("idea_a", "idea_b", "idea_c"),
        ineligible=(
            IneligibleCandidate(
                candidate_id="idea_d",
                assessment_id="paa_d",
                verdict=PriorArtVerdict.NOVELTY_UNRESOLVED,
                reasons=(
                    PriorArtReason(
                        PriorArtReasonCode.SCREENING_TRUNCATED, "truncated"
                    ),
                ),
                overlapping_work_ids=(),
            ),
        ),
        disqualified_candidate_ids=("idea_c",),
        reviews=(
            _review("idea_a"),
            _review("idea_b"),
            _review(
                "idea_c",
                (
                    HardDisqualifier(
                        ground=(
                            DisqualificationGround.RESOURCES_EXCEED_DIRECTIVE
                        ),
                        dimension=DisqualifierDimension.COMPUTE,
                        candidate_text="needs a 64-GPU cluster",
                        constraint_text="One CPU workstation.",
                        why_unrepairable="the scale is the contribution",
                    ),
                ),
            ),
        ),
        pairwise_comparisons=(
            PairwiseComparison(
                first_candidate_id="idea_a",
                second_candidate_id="idea_b",
                comparison="a is cheaper to falsify",
            ),
            PairwiseComparison(
                first_candidate_id="idea_a",
                second_candidate_id="idea_c",
                comparison="a fits the directive, c does not",
            ),
            PairwiseComparison(
                first_candidate_id="idea_b",
                second_candidate_id="idea_c",
                comparison="b fits the directive, c does not",
            ),
        ),
        review_provenance=_provenance(),
        outcome=SelectionOutcome.SELECTED,
        decision=SelectionDecision(
            selected_candidate_id="idea_a",
            decisive_tradeoff="sharper falsifier at equal cost",
            why_selected_over=(
                SelectionRationale(
                    candidate_id="idea_b", reason="undisqualified rival lost"
                ),
            ),
            first_experimental_objective="reproduce the baseline",
            required_capabilities=("dataset download",),
            residual_risks=("dataset-specific effect",),
            provenance=_provenance(),
        ),
        model_calls=2,
        input_tokens=900,
        output_tokens=tokens,
    )


def _no_eligible_run(run_id: str = "sel_2") -> SelectionRunRecord:
    return SelectionRunRecord(
        run_id=run_id,
        directive_id=_directive().id,
        prior_art_run_record_id="prun_1",
        prior_art_run_id="pac_1",
        ideation_run_record_id="irun_1",
        ideation_run_id="idg_1",
        direction_id="dir_1",
        candidate_ids=("idea_a",),
        prior_art_assessment_ids=("paa_a",),
        eligible_candidate_ids=(),
        ineligible=(
            IneligibleCandidate(
                candidate_id="idea_a",
                assessment_id="paa_a",
                verdict=PriorArtVerdict.OVERLAPPING,
                reasons=(),
                overlapping_work_ids=("lit_9",),
            ),
        ),
        disqualified_candidate_ids=(),
        reviews=(),
        pairwise_comparisons=(),
        review_provenance=None,
        outcome=SelectionOutcome.NO_ELIGIBLE_CANDIDATE,
        decision=None,
        model_calls=0,
        input_tokens=0,
        output_tokens=0,
    )


def test_records_round_trip_through_the_store(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    directive = store.record_directive(_directive())
    selected = store.record_run(_selected_run())
    stopped = store.record_run(_no_eligible_run())
    assert store.get_directive(directive.id) == directive
    assert store.get_run(selected.id) == selected
    assert store.get_run(stopped.id) == stopped


def test_an_identical_rewrite_is_idempotent(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    record = store.record_run(_selected_run())
    assert store.record_run(_selected_run()) == record
    assert len(store.runs()) == 1


def test_a_conflicting_rewrite_is_refused(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    record = store.record_run(_selected_run())
    path = tmp_path / "selection" / "runs" / f"{record.id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_calls"] = 3
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(SelectionConflictError, match="never rewritten"):
        store.record_run(_selected_run())


def test_a_tampered_record_is_loud_on_reload(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    record = store.record_run(_selected_run())
    path = tmp_path / "selection" / "runs" / f"{record.id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Still a constructible record — the doctored review reads as honest
    # prose — so the only thing that can catch it is the identity check.
    payload["reviews"][0]["scientific_importance"] = "doctored praise"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    fresh = SelectionStore(tmp_path / "selection")
    with pytest.raises(SelectionIntegrityError, match="no longer matches"):
        fresh.get_run(record.id)


def test_missing_records_read_as_none(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    assert store.get_directive("sdir_missing") is None
    assert store.get_run("srun_missing") is None
    assert store.runs() == ()
    assert store.rejected() == ()


def test_one_record_per_selection_run(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    store.record_run(_selected_run())
    with pytest.raises(SelectionConflictError, match="already recorded"):
        store.record_run(_selected_run(tokens=401))
    # A rerun over the same portfolio is a new occurrence, not a conflict:
    # unselected candidates stay available to future selection runs.
    rerun = store.record_run(_selected_run(run_id="sel_9"))
    assert {record.run_id for record in store.runs()} == {"sel_1", "sel_9"}
    assert store.get_run(rerun.id) == rerun


def test_rejected_payloads_are_preserved_verbatim(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    store.preserve_rejected(
        run_id="sel_1",
        stage="comparative_review",
        reasons=(("misstated_verdict", "the assessment says distinguished"),),
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        payload={"reviews": []},
        repair=0,
    )
    (rejected,) = store.rejected()
    assert rejected["stage"] == "comparative_review"
    assert rejected["reasons"] == [
        {
            "rule": "misstated_verdict",
            "detail": "the assessment says distinguished",
        }
    ]
    assert rejected["payload"] == {"reviews": []}


def test_a_rejection_never_blocks_the_eventual_record(tmp_path: Path) -> None:
    store = SelectionStore(tmp_path / "selection")
    store.preserve_rejected(
        run_id="sel_1",
        stage="selection_decision",
        reasons=(("unknown_candidate", "idea_zz is not in the review"),),
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        payload="not json",
        repair=1,
    )
    record = store.record_run(_selected_run())
    assert store.get_run(record.id) == record
    assert len(store.rejected()) == 1


def test_the_store_lists_records_in_deterministic_order(
    tmp_path: Path,
) -> None:
    store = SelectionStore(tmp_path / "selection")
    first = store.record_run(_selected_run())
    second = store.record_run(_no_eligible_run())
    listed = SelectionStore(tmp_path / "selection").runs()
    assert listed == tuple(
        sorted((first, second), key=lambda record: record.id)
    )
