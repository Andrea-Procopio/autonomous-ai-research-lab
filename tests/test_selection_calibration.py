"""Selection calibration: every outcome stays demonstrably reachable at
the default ceilings, stopping cannot be bought while a defensible
candidate remains, and the outcome boundary is exact. The 5D.2 lesson
applied from day one: a stop that is not forced by grounded evidence is
a gate rejection, not a safe default."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import pytest

from autonomous_research_lab.ideation.direction import DirectionRecord
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
from autonomous_research_lab.runtime.metrics import NO_USAGE
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    UsageLedger,
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
from autonomous_research_lab.selection.selector import (
    CandidateSelector,
    SelectionRejectedError,
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


def _candidate(title: str) -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id="idg_1",
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
                status=DataStatus.NEW_REQUIREMENT,
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


class _Wired:
    """One wired selection setup over freshly recorded upstream runs,
    at the default directive ceilings."""

    def __init__(
        self, tmp_path: Path, verdicts: tuple[PriorArtVerdict, ...]
    ) -> None:
        self.ideation_store = IdeationStore(tmp_path / "ideation")
        self.prior_art_store = PriorArtStore(tmp_path / "priorart")
        direction = self.ideation_store.record_direction(
            DirectionRecord(
                run_id="idg_1",
                snapshot_id="cfp_1",
                scope="mechanistic accounts of in-context learning",
                topics=("adaptation",),
                constraints=("empirical papers",),
                relevant_dates=("2026-01-31",),
                provenance=_provenance(),
            )
        )
        titles = ("Alpha", "Beta", "Gamma", "Delta")
        candidates = tuple(
            self.ideation_store.record_idea(_candidate(titles[index]))
            for index in range(len(verdicts))
        )
        self.candidate_ids = tuple(c.id for c in candidates)
        ideation_run = self.ideation_store.record_run(
            IdeationRunRecord(
                run_id="idg_1",
                directive_id="idir_1",
                assessment_id="madq_1",
                map_run_id="map_1",
                snapshot_id="cfp_1",
                direction_id=direction.id,
                candidate_ids=self.candidate_ids,
                refusal_justification="",
                diversity_rationale="a small portfolio",
                model_calls=2,
                input_tokens=100,
                output_tokens=50,
                portfolio=PortfolioReport(
                    problems_total=1,
                    problems_addressed=1,
                    problems_unaddressed=0,
                    unaddressed_statements=(),
                    addressed_multi_source=0,
                    addressed_tentative=1,
                    addressed_single_source_limitation=0,
                    addressed_contradicted=0,
                    candidates=len(verdicts),
                    distinct_sources_cited=1,
                    themes_targeted=1,
                    distinct_problem_sets=1,
                    distinct_theme_sets=1,
                    distinct_dataset_sets=1,
                    distinct_metric_sets=1,
                ),
            )
        )
        assessments = tuple(
            self.prior_art_store.record_prior_art_assessment(
                _assessment(candidate_id, verdict)
            )
            for candidate_id, verdict in zip(
                self.candidate_ids, verdicts, strict=True
            )
        )
        self.record = self.prior_art_store.record_run(
            PriorArtRunRecord(
                run_id="pac_1",
                directive_id="pdir_1",
                ideation_run_record_id=ideation_run.id,
                ideation_run_id="idg_1",
                assessment_id="madq_1",
                map_run_id="map_1",
                snapshot_id="cfp_1",
                candidate_ids=self.candidate_ids,
                prior_art_assessment_ids=tuple(a.id for a in assessments),
                query_execution_ids=(),
                screening_ids=(),
                comparison_ids=(),
                model_calls=3,
                input_tokens=300,
                output_tokens=150,
            )
        )
        # The default directive: default ceilings, nothing tuned.
        self.directive = SelectionDirective(
            prior_art_run_record_id=self.record.id,
            compute_constraint="One CPU workstation.",
            data_constraint="Public datasets only.",
            time_constraint="Runs finish within hours.",
            experimental_constraint="Containerized seeded runs.",
        )
        self.ledger = UsageLedger()
        self.store = SelectionStore(tmp_path / "selection")

    def selector(
        self, replies: Sequence[str]
    ) -> tuple[CandidateSelector, FakeModelProvider]:
        provider = FakeModelProvider(replies)
        return (
            CandidateSelector(
                provider=provider,
                model="model-x",
                ledger=self.ledger,
                ideation_store=self.ideation_store,
                prior_art_store=self.prior_art_store,
                store=self.store,
            ),
            provider,
        )


_DISQUALIFIER = {
    "ground": "resources_exceed_directive",
    "dimension": "compute",
    "candidate_text": "one GPU-day",
    "constraint_text": "One CPU workstation.",
    "why_unrepairable": (
        "the stated compute need is the contribution's scale"
    ),
}


def _review_reply(
    candidate_ids: Sequence[str],
    disqualify: Sequence[str] = (),
    extra_reviews: Sequence[dict[str, object]] = (),
) -> str:
    reviews: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        entry: dict[str, object] = {
            "candidate_id": candidate_id,
            "prior_art_verdict": "distinguished",
            "disqualifiers": (
                [dict(_DISQUALIFIER)]
                if candidate_id in set(disqualify)
                else []
            ),
        }
        for name in REVIEW_FIELDS:
            entry[name] = f"{name} weighed plainly for {candidate_id}"
        reviews.append(entry)
    reviews.extend(dict(entry) for entry in extra_reviews)
    return json.dumps(
        {
            "reviews": reviews,
            "pairwise_comparisons": [
                {
                    "first_candidate_id": first,
                    "second_candidate_id": second,
                    "comparison": f"{first} and {second} differ in cost",
                }
                for first, second in combinations(
                    sorted(candidate_ids), 2
                )
            ],
        }
    )


def _decision_reply(winner: str, others: Sequence[str]) -> str:
    return json.dumps(
        {
            "selected_candidate_id": winner,
            "decisive_tradeoff": "the cheaper falsifier wins",
            "why_selected_over": [
                {
                    "candidate_id": other,
                    "reason": f"{winner} answers its question sooner",
                }
                for other in others
            ],
            "first_experimental_objective": (
                "reproduce the probe baseline"
            ),
            "required_capabilities": ["dataset download"],
            "residual_risks": ["the effect may be dataset-specific"],
        }
    )


# -- every outcome is reachable at the default ceilings -----------------------------


def test_selected_is_reachable_at_default_ceilings(tmp_path: Path) -> None:
    wired = _Wired(tmp_path, (DISTINGUISHED,) * 3)
    ids = wired.candidate_ids
    selector, _ = wired.selector(
        (_review_reply(ids), _decision_reply(ids[0], list(ids[1:])))
    )
    record = selector.run(wired.directive).run_record
    assert record.outcome is SelectionOutcome.SELECTED


def test_no_eligible_is_reachable_at_default_ceilings(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, (UNRESOLVED, UNRESOLVED))
    selector, provider = wired.selector(())
    record = selector.run(wired.directive).run_record
    assert record.outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE
    assert provider.calls == ()
    assert wired.ledger.drain() == NO_USAGE


def test_no_defensible_is_reachable_at_default_ceilings(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, (DISTINGUISHED, DISTINGUISHED))
    ids = wired.candidate_ids
    selector, provider = wired.selector(
        (_review_reply(ids, disqualify=ids),)
    )
    record = selector.run(wired.directive).run_record
    assert record.outcome is SelectionOutcome.NO_DEFENSIBLE_CANDIDATE
    assert len(provider.calls) == 1


# -- stopping is never the path of least resistance ----------------------------------


def test_selected_survives_a_disqualified_sibling(tmp_path: Path) -> None:
    """One validated disqualifier does not become a stop: the remaining
    contender must still be chosen."""
    wired = _Wired(tmp_path, (DISTINGUISHED, DISTINGUISHED))
    survivor, sibling = wired.candidate_ids
    selector, provider = wired.selector(
        (
            _review_reply(wired.candidate_ids, disqualify=(sibling,)),
            _decision_reply(survivor, []),
        )
    )
    record = selector.run(wired.directive).run_record
    assert record.outcome is SelectionOutcome.SELECTED
    assert record.decision is not None
    assert record.decision.selected_candidate_id == survivor
    assert record.disqualified_candidate_ids == (sibling,)
    assert len(provider.calls) == 2


def test_stopping_cannot_be_bought_while_a_defensible_one_remains(
    tmp_path: Path,
) -> None:
    """A stop-shaped stage-2 reply is a schema violation, and choosing
    the settled sibling is a gate rejection: the corrective call is
    spent, the run fails closed, and no stop record ever exists."""
    wired = _Wired(tmp_path, (DISTINGUISHED, DISTINGUISHED))
    survivor, sibling = wired.candidate_ids
    stop_shaped = json.dumps({"outcome": "no_defensible_candidate"})
    selector, _ = wired.selector(
        (
            _review_reply(wired.candidate_ids, disqualify=(sibling,)),
            stop_shaped,
            _decision_reply(sibling, [survivor]),
        )
    )
    with pytest.raises(
        SelectionRejectedError, match="disqualified_selection"
    ):
        selector.run(wired.directive)
    assert wired.store.runs() == ()
    stages = [str(entry["stage"]) for entry in wired.store.rejected()]
    assert stages.count("selection_decision") == 2


def test_padding_the_review_cannot_buy_the_outcome(tmp_path: Path) -> None:
    """Extra invented reviews are rejected, and after the corrective
    call the stamps are exactly the trusted computation — nothing about
    the outcome is bought by volume."""
    wired = _Wired(tmp_path, (DISTINGUISHED, DISTINGUISHED))
    ids = wired.candidate_ids
    padded_entry: dict[str, object] = {
        "candidate_id": "idea_padded",
        "prior_art_verdict": "distinguished",
        "disqualifiers": [],
    }
    for name in REVIEW_FIELDS:
        padded_entry[name] = "padding"
    selector, provider = wired.selector(
        (
            _review_reply(ids, extra_reviews=(padded_entry,)),
            _review_reply(ids),
            _decision_reply(ids[0], [ids[1]]),
        )
    )
    record = selector.run(wired.directive).run_record
    assert record.outcome is SelectionOutcome.SELECTED
    assert record.eligible_candidate_ids == ids
    assert record.disqualified_candidate_ids == ()
    assert len(provider.calls) == 3
    assert record.review_provenance is not None
    assert record.review_provenance.repair_count == 1


def test_a_misstated_verdict_cannot_create_or_destroy_eligibility(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, (DISTINGUISHED, DISTINGUISHED))
    ids = wired.candidate_ids
    misstated = _review_reply(ids).replace(
        '"prior_art_verdict": "distinguished"',
        '"prior_art_verdict": "novelty_unresolved"',
        1,
    )
    selector, _ = wired.selector(
        (
            misstated,
            _review_reply(ids),
            _decision_reply(ids[0], [ids[1]]),
        )
    )
    record = selector.run(wired.directive).run_record
    assert record.eligible_candidate_ids == ids
    rules: set[str] = set()
    for entry in wired.store.rejected():
        reasons = entry["reasons"]
        assert isinstance(reasons, list)
        for reason in reasons:
            assert isinstance(reason, dict)
            rules.add(str(reason["rule"]))
    assert "misstated_verdict" in rules


def test_unselected_distinguished_candidates_remain_retrievable(
    tmp_path: Path,
) -> None:
    """Not being selected is not a disqualification: the unselected
    stay addressable, and a later selection run over the same portfolio
    is a new occurrence that can choose differently."""
    wired = _Wired(tmp_path, (DISTINGUISHED,) * 3)
    ids = wired.candidate_ids
    first_selector, _ = wired.selector(
        (_review_reply(ids), _decision_reply(ids[0], list(ids[1:])))
    )
    first = first_selector.run(wired.directive).run_record
    assert first.decision is not None
    unselected = {
        entry.candidate_id for entry in first.decision.why_selected_over
    }
    assert unselected == set(ids[1:])
    for candidate_id in unselected:
        assert wired.ideation_store.get_idea(candidate_id) is not None
        assessment = wired.prior_art_store.assessment_for_candidate(
            "pac_1", candidate_id
        )
        assert assessment is not None
        assert assessment.verdict is DISTINGUISHED
    second_selector, _ = wired.selector(
        (_review_reply(ids), _decision_reply(ids[1], [ids[0], ids[2]]))
    )
    second = second_selector.run(wired.directive).run_record
    assert second.decision is not None
    assert second.decision.selected_candidate_id == ids[1]
    assert {record.run_id for record in wired.store.runs()} == {
        first.run_id,
        second.run_id,
    }


# -- the outcome boundary, pure -------------------------------------------------------


def _review(
    candidate_id: str, disqualified: bool = False
) -> CandidateReview:
    prose = {name: f"{name} judged" for name in REVIEW_FIELDS}
    disqualifiers = (
        (
            HardDisqualifier(
                ground=DisqualificationGround.RESOURCES_EXCEED_DIRECTIVE,
                dimension=DisqualifierDimension.COMPUTE,
                candidate_text="one GPU-day",
                constraint_text="One CPU workstation.",
                why_unrepairable="the scale is the contribution",
            ),
        )
        if disqualified
        else ()
    )
    return CandidateReview(
        candidate_id=candidate_id,
        prior_art_verdict=DISTINGUISHED,
        disqualifiers=disqualifiers,
        **prose,
    )


def _record_shape(
    *,
    eligible: tuple[str, ...],
    disqualified: tuple[str, ...],
    outcome: SelectionOutcome,
    winner: str | None,
) -> SelectionRunRecord:
    all_ids = ("idea_a", "idea_b")
    ineligible = tuple(
        IneligibleCandidate(
            candidate_id=candidate_id,
            assessment_id=f"paa_{candidate_id}",
            verdict=UNRESOLVED,
            reasons=(
                PriorArtReason(
                    PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES, "thin"
                ),
            ),
            overlapping_work_ids=(),
        )
        for candidate_id in all_ids
        if candidate_id not in set(eligible)
    )
    reviews = tuple(
        _review(candidate_id, candidate_id in set(disqualified))
        for candidate_id in eligible
    )
    pairs = tuple(
        PairwiseComparison(
            first_candidate_id=first,
            second_candidate_id=second,
            comparison="they differ",
        )
        for first, second in combinations(sorted(eligible), 2)
    )
    decision = (
        SelectionDecision(
            selected_candidate_id=winner,
            decisive_tradeoff="cheaper falsifier",
            why_selected_over=tuple(
                SelectionRationale(candidate_id=other, reason="slower")
                for other in eligible
                if other != winner and other not in set(disqualified)
            ),
            first_experimental_objective="reproduce the baseline",
            required_capabilities=("dataset download",),
            residual_risks=("dataset-specific effect",),
            provenance=_provenance(),
        )
        if winner is not None
        else None
    )
    calls = 0 if outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE else 2
    return SelectionRunRecord(
        run_id="sel_1",
        directive_id="sdir_1",
        prior_art_run_record_id="prun_1",
        prior_art_run_id="pac_1",
        ideation_run_record_id="irun_1",
        ideation_run_id="idg_1",
        direction_id="dir_1",
        candidate_ids=all_ids,
        prior_art_assessment_ids=("paa_idea_a", "paa_idea_b"),
        eligible_candidate_ids=eligible,
        ineligible=ineligible,
        disqualified_candidate_ids=disqualified,
        reviews=reviews,
        pairwise_comparisons=pairs,
        review_provenance=_provenance() if eligible else None,
        outcome=outcome,
        decision=decision,
        model_calls=calls,
        input_tokens=0 if not eligible else 100,
        output_tokens=0 if not eligible else 50,
    )


def test_the_outcome_boundary_is_exact() -> None:
    # 0 eligible: only NO_ELIGIBLE_CANDIDATE is constructible.
    _record_shape(
        eligible=(),
        disqualified=(),
        outcome=SelectionOutcome.NO_ELIGIBLE_CANDIDATE,
        winner=None,
    )
    with pytest.raises(ValueError):
        _record_shape(
            eligible=(),
            disqualified=(),
            outcome=SelectionOutcome.NO_DEFENSIBLE_CANDIDATE,
            winner=None,
        )
    # n of n disqualified: only NO_DEFENSIBLE_CANDIDATE is constructible.
    _record_shape(
        eligible=("idea_a", "idea_b"),
        disqualified=("idea_a", "idea_b"),
        outcome=SelectionOutcome.NO_DEFENSIBLE_CANDIDATE,
        winner=None,
    )
    with pytest.raises(ValueError, match="no validated"):
        _record_shape(
            eligible=("idea_a", "idea_b"),
            disqualified=("idea_a", "idea_b"),
            outcome=SelectionOutcome.SELECTED,
            winner="idea_a",
        )
    # n-1 of n disqualified: a stop cannot be recorded; selection can.
    with pytest.raises(ValueError, match="defensible candidate remains"):
        _record_shape(
            eligible=("idea_a", "idea_b"),
            disqualified=("idea_a",),
            outcome=SelectionOutcome.NO_DEFENSIBLE_CANDIDATE,
            winner=None,
        )
    record = _record_shape(
        eligible=("idea_a", "idea_b"),
        disqualified=("idea_a",),
        outcome=SelectionOutcome.SELECTED,
        winner="idea_b",
    )
    assert record.outcome is SelectionOutcome.SELECTED
