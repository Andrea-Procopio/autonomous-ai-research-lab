"""The admission door: one named SELECTED run, one verified lineage.

Every test wires real stores. The mismatch family is the point of the
door: each test doctors exactly one cross-record fact into a record that
is still individually valid (content ids re-derive cleanly) and proves
the door refuses by comparing records with each other — the seam a
self-consistent forgery would otherwise slip through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_research_lab.admission import (
    AdmissionInputs,
    AdmissionRefusedError,
    require_selected_candidate_for_admission,
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
    PriorArtReason,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
)
from autonomous_research_lab.priorart.records import (
    PriorArtCoverage,
    PriorArtQueryFamily,
    PriorArtRunRecord,
)
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.selection.directive import SelectionDirective
from autonomous_research_lab.selection.records import (
    REVIEW_FIELDS,
    CandidateReview,
    IneligibleCandidate,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRationale,
    SelectionRunRecord,
)
from autonomous_research_lab.selection.store import SelectionStore

DISTINGUISHED = PriorArtVerdict.DISTINGUISHED
UNRESOLVED = PriorArtVerdict.NOVELTY_UNRESOLVED


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


def _candidate(title: str, run_id: str = "idg_1") -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id=run_id,
        title=title,
        research_question=f"{title}?",
        proposed_contribution="a causal test of head reweighting",
        mechanism="reweighting amplifies specialized induction heads",
        hypothesis="reweighted heads carry the in-context ability",
        grounding="the cited records report head specialization",
        predictions=(
            Prediction(
                text="ablating reweighted heads drops accuracy",
                falsifier="accuracy unchanged after ablation",
            ),
        ),
        datasets=(
            DataRequirement(
                name="synthetic sequences",
                status=DataStatus.EXISTING,
                role="probe tasks",
            ),
        ),
        metrics=("accuracy",),
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


def _portfolio(candidates: int) -> PortfolioReport:
    return PortfolioReport(
        problems_total=1,
        problems_addressed=1,
        problems_unaddressed=0,
        unaddressed_statements=(),
        addressed_multi_source=0,
        addressed_tentative=1,
        addressed_single_source_limitation=0,
        addressed_contradicted=0,
        candidates=candidates,
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


def _assessment(
    candidate_id: str, verdict: PriorArtVerdict
) -> PriorArtAssessment:
    reasons = (
        (
            PriorArtReason(
                PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES, "thin pool"
            ),
        )
        if verdict is UNRESOLVED
        else ()
    )
    return PriorArtAssessment(
        run_id="pac_1",
        candidate_id=candidate_id,
        directive_id="pdir_1",
        verdict=verdict,
        overlapping_work_ids=(),
        compared_work_ids=("lit_1", "lit_2"),
        reasons=reasons,
        thresholds=PriorArtThresholds(),
        coverage=_coverage(),
    )


def _review(candidate_id: str) -> CandidateReview:
    prose = {name: f"{name} for {candidate_id}" for name in REVIEW_FIELDS}
    return CandidateReview(
        candidate_id=candidate_id,
        prior_art_verdict=DISTINGUISHED,
        disqualifiers=(),
        **prose,
    )


class _Lineage:
    """One fully wired lineage: snapshot, direction, candidates, runs,
    assessments, and a valid SELECTED selection run over them."""

    def __init__(
        self,
        tmp_path: Path,
        verdicts: tuple[PriorArtVerdict, ...] = (
            DISTINGUISHED,
            DISTINGUISHED,
        ),
        candidate_run_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.ideation_store = IdeationStore(tmp_path / "ideation")
        self.prior_art_store = PriorArtStore(tmp_path / "priorart")
        self.selection_store = SelectionStore(tmp_path / "selection")
        self.snapshot = self.ideation_store.record_snapshot(
            CfpSnapshot(
                source_url="https://example.org/cfp",
                supplied_at="2026-08-01T00:00:00",
                text="A workshop on in-context learning mechanisms.",
            )
        )
        self.direction = self.ideation_store.record_direction(
            DirectionRecord(
                run_id="idg_1",
                snapshot_id=self.snapshot.id,
                scope="mechanistic accounts of in-context learning",
                topics=("adaptation",),
                constraints=("empirical papers",),
                relevant_dates=("2026-01-31",),
                provenance=_provenance(),
            )
        )
        titles = ("Alpha", "Beta", "Gamma", "Delta")
        run_ids = candidate_run_ids or ("idg_1",) * len(verdicts)
        self.candidates = tuple(
            self.ideation_store.record_idea(
                _candidate(titles[index], run_ids[index])
            )
            for index in range(len(verdicts))
        )
        self.candidate_ids = tuple(c.id for c in self.candidates)
        self.ideation_run = self.ideation_store.record_run(
            IdeationRunRecord(
                run_id="idg_1",
                directive_id="idir_1",
                assessment_id="madq_1",
                map_run_id="map_1",
                snapshot_id=self.snapshot.id,
                direction_id=self.direction.id,
                candidate_ids=self.candidate_ids,
                refusal_justification="",
                diversity_rationale="a small portfolio",
                model_calls=2,
                input_tokens=100,
                output_tokens=50,
                portfolio=_portfolio(len(verdicts)),
            )
        )
        self.assessments = tuple(
            self.prior_art_store.record_prior_art_assessment(
                _assessment(candidate_id, verdict)
            )
            for candidate_id, verdict in zip(
                self.candidate_ids, verdicts, strict=True
            )
        )
        self.prior_art_run = self.prior_art_store.record_run(
            PriorArtRunRecord(
                run_id="pac_1",
                directive_id="pdir_1",
                ideation_run_record_id=self.ideation_run.id,
                ideation_run_id="idg_1",
                assessment_id="madq_1",
                map_run_id="map_1",
                snapshot_id=self.snapshot.id,
                candidate_ids=self.candidate_ids,
                prior_art_assessment_ids=tuple(
                    a.id for a in self.assessments
                ),
                query_execution_ids=(),
                screening_ids=(),
                comparison_ids=(),
                model_calls=3,
                input_tokens=300,
                output_tokens=150,
            )
        )
        self.eligible_ids = tuple(
            candidate_id
            for candidate_id, verdict in zip(
                self.candidate_ids, verdicts, strict=True
            )
            if verdict is DISTINGUISHED
        )
        self.directive = self.selection_store.record_directive(
            SelectionDirective(
                prior_art_run_record_id=self.prior_art_run.id,
                compute_constraint="One CPU workstation.",
                data_constraint="Public datasets only.",
                time_constraint="Runs finish within hours.",
                experimental_constraint="Containerized seeded runs.",
            )
        )

    def selection_record(self, **overrides: object) -> SelectionRunRecord:
        """A valid SELECTED record over the wired lineage, with any
        field doctored by the caller — individually consistent, so only
        a cross-record check can catch the doctoring."""
        from itertools import combinations

        ineligible = tuple(
            entry
            for entry in _partition_ineligible(self)
            if entry.candidate_id not in self.eligible_ids
        )
        decision = None
        if self.eligible_ids:
            winner = self.eligible_ids[0]
            decision = SelectionDecision(
                selected_candidate_id=winner,
                decisive_tradeoff="sharper falsifier at equal cost",
                why_selected_over=tuple(
                    SelectionRationale(
                        candidate_id=candidate_id,
                        reason="the rival lost on diagnosticity",
                    )
                    for candidate_id in self.eligible_ids
                    if candidate_id != winner
                ),
                first_experimental_objective="reproduce the baseline",
                required_capabilities=("dataset download",),
                residual_risks=("dataset-specific effect",),
                provenance=_provenance(),
            )
        values: dict[str, object] = {
            "run_id": "sel_1",
            "directive_id": self.directive.id,
            "prior_art_run_record_id": self.prior_art_run.id,
            "prior_art_run_id": self.prior_art_run.run_id,
            "ideation_run_record_id": self.ideation_run.id,
            "ideation_run_id": self.ideation_run.run_id,
            "direction_id": self.direction.id,
            "candidate_ids": self.candidate_ids,
            "prior_art_assessment_ids": tuple(
                a.id for a in self.assessments
            ),
            "eligible_candidate_ids": self.eligible_ids,
            "ineligible": ineligible,
            "disqualified_candidate_ids": (),
            "reviews": tuple(
                _review(candidate_id) for candidate_id in self.eligible_ids
            ),
            "pairwise_comparisons": tuple(
                PairwiseComparison(
                    first_candidate_id=first,
                    second_candidate_id=second,
                    comparison="a grounded comparison",
                )
                for first, second in combinations(
                    sorted(self.eligible_ids), 2
                )
            ),
            "review_provenance": (
                _provenance() if self.eligible_ids else None
            ),
            "outcome": (
                SelectionOutcome.SELECTED
                if self.eligible_ids
                else SelectionOutcome.NO_ELIGIBLE_CANDIDATE
            ),
            "decision": decision,
            "model_calls": 2 if self.eligible_ids else 0,
            "input_tokens": 900 if self.eligible_ids else 0,
            "output_tokens": 400 if self.eligible_ids else 0,
        }
        values.update(overrides)
        return SelectionRunRecord(**values)  # type: ignore[arg-type]

    def record(self, **overrides: object) -> SelectionRunRecord:
        return self.selection_store.record_run(
            self.selection_record(**overrides)
        )

    def door(self, record_id: str) -> AdmissionInputs:
        return require_selected_candidate_for_admission(
            self.selection_store,
            self.prior_art_store,
            self.ideation_store,
            record_id,
        )


def _partition_ineligible(
    lineage: _Lineage,
) -> tuple[IneligibleCandidate, ...]:
    entries = []
    for candidate_id, assessment in zip(
        lineage.candidate_ids, lineage.assessments, strict=True
    ):
        if assessment.verdict is DISTINGUISHED:
            continue
        entries.append(
            IneligibleCandidate(
                candidate_id=candidate_id,
                assessment_id=assessment.id,
                verdict=assessment.verdict,
                reasons=assessment.reasons,
                overlapping_work_ids=assessment.overlapping_work_ids,
            )
        )
    return tuple(entries)


def test_a_selected_run_loads_its_whole_lineage(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path, (DISTINGUISHED, DISTINGUISHED, UNRESOLVED))
    record = lineage.record()

    inputs = lineage.door(record.id)

    assert inputs.selection_run == record
    assert inputs.selected.id == lineage.eligible_ids[0]
    assert inputs.selected_assessment.candidate_id == (
        lineage.eligible_ids[0]
    )
    assert inputs.snapshot == lineage.snapshot
    assert inputs.selection_directive == lineage.directive


def test_a_missing_selection_run_refuses_by_name(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path)
    with pytest.raises(AdmissionRefusedError, match="srun_missing"):
        lineage.door("srun_missing")


def test_each_stop_outcome_refuses_before_any_call(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path, (UNRESOLVED, UNRESOLVED))
    stop = lineage.record(run_id="sel_stop")
    with pytest.raises(
        AdmissionRefusedError, match="no_eligible_candidate"
    ) as caught:
        lineage.door(stop.id)
    assert "honest stop" in str(caught.value)


def test_a_missing_selection_directive_refuses(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path)
    record = lineage.record(directive_id="sdir_0000000000000bad")
    with pytest.raises(AdmissionRefusedError, match="sdir_0000000000000bad"):
        lineage.door(record.id)


def test_a_missing_upstream_lineage_refuses_as_admission(
    tmp_path: Path,
) -> None:
    """The reused selection door's refusal is re-raised as the admission
    error, naming the admission context."""
    lineage = _Lineage(tmp_path)
    record = lineage.record(
        prior_art_run_record_id="prun_0000000000000bad"
    )
    with pytest.raises(AdmissionRefusedError, match="admission of") as caught:
        lineage.door(record.id)
    assert "prun_0000000000000bad" in str(caught.value)


def test_every_cross_record_disagreement_refuses(tmp_path: Path) -> None:
    """One doctored fact per case; each record stays individually valid,
    so only the cross-record equality can catch it."""
    lineage = _Lineage(tmp_path)
    cases: tuple[tuple[str, dict[str, object]], ...] = (
        ("prior_art_run_id", {"prior_art_run_id": "pac_2"}),
        ("ideation_run_id", {"ideation_run_id": "idg_2"}),
        ("direction_id", {"direction_id": "dir_0000000000000bad"}),
    )
    for index, (label, overrides) in enumerate(cases):
        record = lineage.record(run_id=f"sel_bad_{index}", **overrides)
        with pytest.raises(AdmissionRefusedError, match=label):
            lineage.door(record.id)


def test_a_forged_portfolio_list_refuses(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path)
    reversed_ids = tuple(reversed(lineage.candidate_ids))
    reversed_assessments = tuple(
        reversed([a.id for a in lineage.assessments])
    )
    record = lineage.record(
        run_id="sel_forged",
        candidate_ids=reversed_ids,
        prior_art_assessment_ids=reversed_assessments,
        eligible_candidate_ids=reversed_ids,
        reviews=tuple(_review(c) for c in reversed_ids),
    )
    with pytest.raises(AdmissionRefusedError, match="different portfolio"):
        lineage.door(record.id)


def test_a_forged_eligible_stamp_refuses(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path)
    reversed_eligible = tuple(reversed(lineage.eligible_ids))
    record = lineage.record(
        run_id="sel_stamped",
        eligible_candidate_ids=reversed_eligible,
        reviews=tuple(_review(c) for c in reversed_eligible),
    )
    with pytest.raises(
        AdmissionRefusedError, match="computed, never copied"
    ):
        lineage.door(record.id)


def test_a_winner_from_another_ideation_run_refuses(tmp_path: Path) -> None:
    lineage = _Lineage(
        tmp_path,
        (DISTINGUISHED, DISTINGUISHED),
        candidate_run_ids=("idg_2", "idg_1"),
    )
    record = lineage.record()
    with pytest.raises(AdmissionRefusedError, match="another run"):
        lineage.door(record.id)


def test_a_missing_snapshot_refuses(tmp_path: Path) -> None:
    lineage = _Lineage(tmp_path)
    record = lineage.record()
    path = (
        tmp_path
        / "ideation"
        / "snapshots"
        / f"{lineage.snapshot.id}.json"
    )
    path.unlink()
    with pytest.raises(AdmissionRefusedError, match=lineage.snapshot.id):
        lineage.door(record.id)


def test_only_the_named_selection_run_is_read(tmp_path: Path) -> None:
    """Two durable selection runs; the door admits from exactly the one
    the caller names, whatever else the store holds."""
    lineage = _Lineage(tmp_path, (DISTINGUISHED, DISTINGUISHED))
    first = lineage.record()
    second_winner = lineage.eligible_ids[1]
    second = lineage.record(
        run_id="sel_2",
        decision=SelectionDecision(
            selected_candidate_id=second_winner,
            decisive_tradeoff="the other tradeoff",
            why_selected_over=(
                SelectionRationale(
                    candidate_id=lineage.eligible_ids[0],
                    reason="the rival lost on feasibility",
                ),
            ),
            first_experimental_objective="reproduce the baseline",
            required_capabilities=("dataset download",),
            residual_risks=("dataset-specific effect",),
            provenance=_provenance(),
        ),
    )

    assert lineage.door(first.id).selected.id == lineage.eligible_ids[0]
    assert lineage.door(second.id).selected.id == second_winner
