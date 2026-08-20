"""Calibration: the live shapes admit at the default wiring, coverage
scales, reduction is refused rather than silent, and no smuggled
candidate or synthetic science has a route in."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import pytest

from autonomous_research_lab.admission import (
    AdmissionDirective,
    AdmissionRecord,
    AdmissionRejectedError,
    AdmissionRunResult,
    AdmissionStore,
    CandidateAdmitter,
)
from autonomous_research_lab.ideation.direction import (
    CfpSnapshot,
    DirectionRecord,
)
from autonomous_research_lab.ideation.records import (
    AddressedProblem,
    CandidateIdea,
    DataRequirement,
    DataStatus,
    IdeationRunRecord,
    PortfolioReport,
    Prediction,
    ResourceEstimate,
    TargetedTheme,
    problem_key,
    theme_key,
)
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    ThemeEra,
)
from autonomous_research_lab.priorart.assessment import (
    PriorArtAssessment,
    PriorArtThresholds,
    PriorArtVerdict,
)
from autonomous_research_lab.priorart.records import (
    PriorArtCoverage,
    PriorArtQueryFamily,
    PriorArtRunRecord,
)
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    ScriptedReply,
    UsageLedger,
)
from autonomous_research_lab.selection.directive import SelectionDirective
from autonomous_research_lab.selection.records import (
    REVIEW_FIELDS,
    CandidateReview,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRunRecord,
)
from autonomous_research_lab.selection.store import SelectionStore

DISTINGUISHED = PriorArtVerdict.DISTINGUISHED

#: The live 5E winner's shape: one comparative prediction whose
#: falsifier names both arms, five declared metrics.
LIVE_METRICS = (
    "few-shot accuracy",
    "prefix matching score",
    "copying score",
    "overlap of important heads",
    "tail-token logit increase",
)
LIVE_PREDICTION = Prediction(
    text=(
        "Top-weighted heads overlap strongly with high induction heads "
        "and ablating them drops accuracy substantially more than "
        "ablating bottom-weighted heads."
    ),
    falsifier=(
        "If ablating top-weighted heads causes similar drop to ablating "
        "bottom-weighted heads, prediction fails."
    ),
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
        repair_count=0,
    )


def _candidate(
    title: str,
    predictions: tuple[Prediction, ...],
    metrics: tuple[str, ...],
) -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id="idg_1",
        title=title,
        research_question=f"{title}?",
        proposed_contribution="a causal test of head reweighting",
        mechanism="reweighting amplifies specialized induction heads",
        hypothesis="reweighted heads carry the in-context ability",
        grounding="the cited records report head specialization",
        predictions=predictions,
        datasets=(
            DataRequirement(
                name="synthetic sequences",
                status=DataStatus.EXISTING,
                role="probe tasks",
            ),
        ),
        metrics=metrics,
        evaluation_protocol="held-out probes",
        baselines=("dense fine-tuning",),
        ablations=("random head subsets",),
        resources=ResourceEstimate(
            compute="one GPU-day", data="synthetic", implementation="small"
        ),
        risks=("effects may not localize",),
        cfp_alignment="matches the adaptation topic",
        aligned_topics=("adaptation",),
        uncertainty="the mechanism may be diffuse",
        search_terms=("attention head reweighting",),
        addressed_problems=(
            AddressedProblem(
                key=problem_key(statement),
                statement=statement,
                kind=ProblemKind.OPEN_PROBLEM,
                tier=SupportTier.TENTATIVE,
            ),
        ),
        targeted_themes=(
            TargetedTheme(
                key=theme_key("adaptation"),
                name="adaptation",
                era=ThemeEra.RECENT,
            ),
        ),
        cited_source_ids=("lit_c1",),
        cited_recent=1,
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _portfolio() -> PortfolioReport:
    return PortfolioReport(
        problems_total=1,
        problems_addressed=1,
        problems_unaddressed=0,
        unaddressed_statements=(),
        addressed_multi_source=0,
        addressed_tentative=1,
        addressed_single_source_limitation=0,
        addressed_contradicted=0,
        candidates=1,
        distinct_sources_cited=1,
        themes_targeted=1,
        distinct_problem_sets=1,
        distinct_theme_sets=1,
        distinct_dataset_sets=1,
        distinct_metric_sets=1,
    )


def _coverage() -> PriorArtCoverage:
    return PriorArtCoverage(
        families_executed=tuple(
            family.value for family in PriorArtQueryFamily
        ),
        queries_executed=6,
        total_retrieved=12,
        unique_sources=12,
        overlap=2,
        saturation=0.5,
        post_cutoff_excluded=0,
        undated_sources=0,
        abstract_level=12,
        metadata_level=0,
        known_prior_art_listed=2,
        known_prior_art_recovered=1,
        screened=12,
        potential_overlap=2,
        related=2,
        unrelated=8,
        undecidable=0,
        metadata_ambiguous=0,
        screening_truncated=0,
        compared_works=2,
    )


class _Wired:
    """A one-candidate wired admission over a configurable candidate."""

    def __init__(
        self,
        tmp_path: Path,
        replies: Sequence[ScriptedReply | str],
        predictions: tuple[Prediction, ...] = (LIVE_PREDICTION,),
        metrics: tuple[str, ...] = LIVE_METRICS,
    ) -> None:
        self.ideation_store = IdeationStore(tmp_path / "ideation")
        self.prior_art_store = PriorArtStore(tmp_path / "priorart")
        self.selection_store = SelectionStore(tmp_path / "selection")
        snapshot = self.ideation_store.record_snapshot(
            CfpSnapshot(
                source_url="https://example.org/cfp",
                supplied_at="2026-08-01T00:00:00",
                text="A workshop on in-context learning mechanisms.",
            )
        )
        direction = self.ideation_store.record_direction(
            DirectionRecord(
                run_id="idg_1",
                snapshot_id=snapshot.id,
                scope="mechanistic accounts of in-context learning",
                topics=("adaptation",),
                constraints=("empirical papers",),
                relevant_dates=("2026-01-31",),
                provenance=_provenance(),
            )
        )
        self.candidate = self.ideation_store.record_idea(
            _candidate("Alpha", predictions, metrics)
        )
        ideation_run = self.ideation_store.record_run(
            IdeationRunRecord(
                run_id="idg_1",
                directive_id="idir_1",
                assessment_id="madq_1",
                map_run_id="map_1",
                snapshot_id=snapshot.id,
                direction_id=direction.id,
                candidate_ids=(self.candidate.id,),
                refusal_justification="",
                diversity_rationale="a single-candidate portfolio",
                model_calls=2,
                input_tokens=100,
                output_tokens=50,
                portfolio=_portfolio(),
            )
        )
        assessment = self.prior_art_store.record_prior_art_assessment(
            PriorArtAssessment(
                run_id="pac_1",
                candidate_id=self.candidate.id,
                directive_id="pdir_1",
                verdict=DISTINGUISHED,
                overlapping_work_ids=(),
                compared_work_ids=("lit_1", "lit_2"),
                reasons=(),
                thresholds=PriorArtThresholds(),
                coverage=_coverage(),
            )
        )
        prior_art_run = self.prior_art_store.record_run(
            PriorArtRunRecord(
                run_id="pac_1",
                directive_id="pdir_1",
                ideation_run_record_id=ideation_run.id,
                ideation_run_id="idg_1",
                assessment_id="madq_1",
                map_run_id="map_1",
                snapshot_id=snapshot.id,
                candidate_ids=(self.candidate.id,),
                prior_art_assessment_ids=(assessment.id,),
                query_execution_ids=(),
                screening_ids=(),
                comparison_ids=(),
                model_calls=3,
                input_tokens=300,
                output_tokens=150,
            )
        )
        selection_directive = self.selection_store.record_directive(
            SelectionDirective(
                prior_art_run_record_id=prior_art_run.id,
                compute_constraint="One CPU workstation.",
                data_constraint="Public datasets only.",
                time_constraint="Runs finish within hours.",
                experimental_constraint="Containerized seeded runs.",
            )
        )
        prose = {
            name: f"{name} for the single candidate"
            for name in REVIEW_FIELDS
        }
        self.selection_run = self.selection_store.record_run(
            SelectionRunRecord(
                run_id="sel_1",
                directive_id=selection_directive.id,
                prior_art_run_record_id=prior_art_run.id,
                prior_art_run_id="pac_1",
                ideation_run_record_id=ideation_run.id,
                ideation_run_id="idg_1",
                direction_id=direction.id,
                candidate_ids=(self.candidate.id,),
                prior_art_assessment_ids=(assessment.id,),
                eligible_candidate_ids=(self.candidate.id,),
                ineligible=(),
                disqualified_candidate_ids=(),
                reviews=(
                    CandidateReview(
                        candidate_id=self.candidate.id,
                        prior_art_verdict=DISTINGUISHED,
                        disqualifiers=(),
                        **prose,
                    ),
                ),
                pairwise_comparisons=tuple(
                    PairwiseComparison(
                        first_candidate_id=first,
                        second_candidate_id=second,
                        comparison="a grounded comparison",
                    )
                    for first, second in combinations(
                        sorted((self.candidate.id,)), 2
                    )
                ),
                review_provenance=_provenance(),
                outcome=SelectionOutcome.SELECTED,
                decision=SelectionDecision(
                    selected_candidate_id=self.candidate.id,
                    decisive_tradeoff="the only defensible contender",
                    why_selected_over=(),
                    first_experimental_objective="reproduce the baseline",
                    required_capabilities=("dataset download",),
                    residual_risks=("dataset-specific effect",),
                    provenance=_provenance(),
                ),
                model_calls=2,
                input_tokens=900,
                output_tokens=400,
            )
        )
        self.directive = AdmissionDirective(
            selection_run_record_id=self.selection_run.id,
            scheduling_requirement="Batch-scheduled execution.",
            job_duration_requirement="Jobs bounded to two days.",
            checkpoint_requirement="Checkpoint and resume required.",
        )
        self.provider = FakeModelProvider(replies)
        self.store = AdmissionStore(tmp_path / "admission")
        self.admitter = CandidateAdmitter(
            provider=self.provider,
            model="model-x",
            ledger=UsageLedger(),
            ideation_store=self.ideation_store,
            prior_art_store=self.prior_art_store,
            selection_store=self.selection_store,
            store=self.store,
        )

    def run(self) -> AdmissionRunResult:
        return self.admitter.run(self.directive)


def _encoding(
    prediction: Prediction,
    base_metric: str,
    higher: str,
    lower: str,
    contrary: str,
    condition: str = "few-shot evaluation over the recorded protocol",
) -> dict[str, object]:
    return {
        "prediction_text": prediction.text,
        "condition": condition,
        "base_metric": base_metric,
        "expected_higher_arm": higher,
        "expected_lower_arm": lower,
        "contrary_observation": contrary,
        "support": [
            {
                "source": "candidate",
                "field_path": "hypothesis",
                "quote": "reweighted heads carry the in-context ability",
            }
        ],
    }


def _live_encoding(**overrides: object) -> dict[str, object]:
    entry = _encoding(
        LIVE_PREDICTION,
        "few-shot accuracy",
        "ablating top-weighted heads",
        "ablating bottom-weighted heads",
        "similar drop to ablating bottom-weighted heads",
    )
    entry.update(overrides)
    return entry


def _reply(*encodings: dict[str, object]) -> str:
    return json.dumps({"operational_predictions": list(encodings)})


def test_the_live_shape_admits_at_the_default_wiring(
    tmp_path: Path,
) -> None:
    """One comparative prediction, five metrics — the exact shape of
    the 5E winner — admits in one call with the default ceilings."""
    wired = _Wired(tmp_path, (_reply(_live_encoding()),))
    result = wired.run()

    assert result.record.model_calls == 1
    assert result.record.measurements == LIVE_METRICS
    (prediction,) = result.state.predictions
    assert prediction.metric == (
        "difference in few-shot accuracy: ablating top-weighted heads "
        "minus ablating bottom-weighted heads"
    )


def test_two_observables_of_one_prediction_both_admit(
    tmp_path: Path,
) -> None:
    """The live prediction carries two observables (overlap AND the
    ablation drop); encoding both is legal within the cap, and each
    derives its own distinct core prediction."""
    second = _live_encoding(
        base_metric="overlap of important heads",
        expected_higher_arm="Top-weighted heads",
        expected_lower_arm="bottom-weighted heads",
        contrary_observation="similar drop to ablating",
    )
    wired = _Wired(tmp_path, (_reply(_live_encoding(), second),))
    result = wired.run()

    assert len(result.record.operational_predictions) == 2
    assert len(result.state.predictions) == 2
    assert len(set(result.record.prediction_ids)) == 2


def test_a_three_prediction_candidate_admits_with_full_coverage(
    tmp_path: Path,
) -> None:
    predictions = tuple(
        Prediction(
            text=f"probe {name} rises under reweighting",
            falsifier=f"probe {name} unchanged under reweighting",
        )
        for name in ("alpha", "beta", "gamma")
    )
    encodings = tuple(
        _encoding(
            prediction,
            "few-shot accuracy",
            f"probe {name} rises",
            f"probe {name} unchanged",
            f"probe {name} unchanged",
        )
        for name, prediction in zip(
            ("alpha", "beta", "gamma"), predictions, strict=True
        )
    )
    wired = _Wired(
        tmp_path, (_reply(*encodings),), predictions=predictions
    )
    result = wired.run()

    assert len(result.state.predictions) == 3
    covered = {
        entry.prediction_text
        for entry in result.record.operational_predictions
    }
    assert covered == {prediction.text for prediction in predictions}


def test_a_skipped_prediction_is_a_rejection_not_a_truncation(
    tmp_path: Path,
) -> None:
    """Covering two of three recorded predictions is not a smaller
    admission — it is no admission. The same incomplete reply on the
    corrective changes nothing."""
    predictions = tuple(
        Prediction(
            text=f"probe {name} rises under reweighting",
            falsifier=f"probe {name} unchanged under reweighting",
        )
        for name in ("alpha", "beta", "gamma")
    )
    partial = _reply(
        *(
            _encoding(
                prediction,
                "few-shot accuracy",
                f"probe {name} rises",
                f"probe {name} unchanged",
                f"probe {name} unchanged",
            )
            for name, prediction in zip(
                ("alpha", "beta"), predictions[:2], strict=True
            )
        )
    )
    wired = _Wired(
        tmp_path, (partial, partial), predictions=predictions
    )
    with pytest.raises(AdmissionRejectedError, match="missing_decision"):
        wired.run()

    assert wired.store.records() == ()
    assert len(wired.store.rejected()) == 2


def test_a_smuggled_candidate_cannot_be_admitted(tmp_path: Path) -> None:
    """Text from outside the selected candidate's record has no route
    in: an unshown id is a fabricated reference, and an unshown quote
    fails support verbatim."""
    smuggled = _live_encoding(
        condition=(
            "the setting idea_00000000000000ff proposes instead"
        ),
        support=[
            {
                "source": "candidate",
                "field_path": "hypothesis",
                "quote": "a sentence no shown record contains",
            }
        ],
    )
    reply = _reply(smuggled)
    wired = _Wired(tmp_path, (reply, reply))
    with pytest.raises(
        AdmissionRejectedError, match="fabricated_reference"
    ) as caught:
        wired.run()

    assert "unsupported_claim" in str(caught.value)
    assert wired.store.records() == ()


def test_no_result_evidence_or_assessment_is_expressible() -> None:
    """Pure and structural: the admission record has no field that
    could hold a result, evidence, an assessment, a claim, or a novelty
    verdict — fabricated science has nowhere to land."""
    # The lineage id fields legitimately contain "assessment" — they
    # name upstream records, never verdicts of admission's own — so the
    # banned fragments are the science-shaped ones.
    banned = (
        "result",
        "evidence",
        "verdict",
        "claim",
        "novelty",
        "manuscript",
    )
    for entry in dataclasses.fields(AdmissionRecord):
        for fragment in banned:
            assert fragment not in entry.name, (entry.name, fragment)
