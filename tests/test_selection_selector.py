"""The candidate selector end to end: deterministic provider fakes,
trusted stamps the model cannot alter, bounded repair, exact accounting,
and upstream records that stay byte-identical."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
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
    NoveltyStatus,
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
from autonomous_research_lab.runtime.metrics import NO_USAGE, ProviderUsage
from autonomous_research_lab.runtime.providers import (
    CallAccounting,
    FakeModelProvider,
    ProviderRateLimitError,
    ScriptedReply,
    StructuredOutputError,
    UsageLedger,
)
from autonomous_research_lab.selection.directive import SelectionDirective
from autonomous_research_lab.selection.records import (
    REVIEW_FIELDS,
    SelectionOutcome,
    SelectionRunRecord,
)
from autonomous_research_lab.selection.selector import (
    COMPARATIVE_REVIEW_INSTRUCTION,
    COMPARATIVE_REVIEW_SCHEMA,
    SELECTION_DECISION_SCHEMA,
    CandidateSelector,
    SelectionRejectedError,
    SelectionRunResult,
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


def _direction() -> DirectionRecord:
    return DirectionRecord(
        run_id="idg_1",
        snapshot_id="cfp_1",
        scope="mechanistic accounts of in-context learning",
        topics=("adaptation",),
        constraints=("empirical papers",),
        relevant_dates=("2026-01-31",),
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
    candidate_id: str, verdict: PriorArtVerdict, run_id: str = "pac_1"
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
        run_id=run_id,
        candidate_id=candidate_id,
        directive_id="pdir_1",
        verdict=verdict,
        overlapping_work_ids=(
            ("lit_1",) if verdict is PriorArtVerdict.OVERLAPPING else ()
        ),
        compared_work_ids=("lit_1", "lit_2"),
        reasons=reasons,
        thresholds=PriorArtThresholds(),
        coverage=_coverage(),
    )


class _Wired:
    """One wired selection setup over freshly recorded upstream runs."""

    def __init__(
        self,
        tmp_path: Path,
        replies: Sequence[ScriptedReply | str],
        verdicts: tuple[PriorArtVerdict, ...],
    ) -> None:
        self.ideation_store = IdeationStore(tmp_path / "ideation")
        self.prior_art_store = PriorArtStore(tmp_path / "priorart")
        direction = self.ideation_store.record_direction(_direction())
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
                portfolio=_portfolio(len(verdicts)),
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
        self.directive = SelectionDirective(
            prior_art_run_record_id=self.record.id,
            compute_constraint="One CPU workstation.",
            data_constraint="Public datasets only.",
            time_constraint="Runs finish within hours.",
            experimental_constraint="Containerized seeded runs.",
        )
        self.provider = FakeModelProvider(replies)
        self.ledger = UsageLedger()
        self.store = SelectionStore(tmp_path / "selection")
        self.selector = CandidateSelector(
            provider=self.provider,
            model="model-x",
            ledger=self.ledger,
            ideation_store=self.ideation_store,
            prior_art_store=self.prior_art_store,
            store=self.store,
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


def _review_entry(
    candidate_id: str, *, disqualify: bool = False
) -> dict[str, object]:
    entry: dict[str, object] = {
        "candidate_id": candidate_id,
        "prior_art_verdict": "distinguished",
        "disqualifiers": [dict(_DISQUALIFIER)] if disqualify else [],
    }
    for name in REVIEW_FIELDS:
        entry[name] = f"{name} weighed plainly for {candidate_id}"
    return entry


def _review_reply(
    candidate_ids: Sequence[str], disqualify: Sequence[str] = ()
) -> str:
    return json.dumps(
        {
            "reviews": [
                _review_entry(
                    candidate_id,
                    disqualify=candidate_id in set(disqualify),
                )
                for candidate_id in candidate_ids
            ],
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


def _happy(wired: _Wired) -> tuple[str, str]:
    ids = wired.candidate_ids
    return (
        _review_reply(ids),
        _decision_reply(ids[0], list(ids[1:])),
    )


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


# -- outcomes -------------------------------------------------------------------


def test_a_defensible_candidate_is_selected(tmp_path: Path) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 3)
    wired.provider = FakeModelProvider(_happy(wired))
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    result = wired.selector.run(wired.directive)
    record = result.run_record
    assert record.outcome is SelectionOutcome.SELECTED
    assert record.decision is not None
    assert record.decision.selected_candidate_id == wired.candidate_ids[0]
    assert len(wired.provider.calls) == 2
    assert record.model_calls == 2
    assert wired.store.get_run(record.id) == record


def _run_happy(
    tmp_path: Path, verdicts: tuple[PriorArtVerdict, ...]
) -> tuple[_Wired, SelectionRunResult]:
    wired = _Wired(tmp_path, [], verdicts)
    eligible = tuple(
        candidate_id
        for candidate_id, verdict in zip(
            wired.candidate_ids, verdicts, strict=True
        )
        if verdict is DISTINGUISHED
    )
    replies = (
        _review_reply(eligible),
        _decision_reply(eligible[0], list(eligible[1:])),
    )
    wired.provider = FakeModelProvider(replies)
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    return wired, wired.selector.run(wired.directive)


def test_exactly_one_candidate_is_ever_selected(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path, (DISTINGUISHED,) * 3)
    record = result.run_record
    assert record.eligible_candidate_ids == wired.candidate_ids
    assert record.decision is not None
    others = {
        entry.candidate_id for entry in record.decision.why_selected_over
    }
    assert others == set(wired.candidate_ids[1:])
    assert record.disqualified_candidate_ids == ()


def test_zero_eligible_candidates_never_call_the_model(
    tmp_path: Path,
) -> None:
    wired = _Wired(
        tmp_path, [], (UNRESOLVED, PriorArtVerdict.OVERLAPPING)
    )
    result = wired.selector.run(wired.directive)
    record = result.run_record
    assert record.outcome is SelectionOutcome.NO_ELIGIBLE_CANDIDATE
    assert wired.provider.calls == ()
    assert wired.ledger.drain() == NO_USAGE
    assert record.model_calls == 0
    assert record.input_tokens == 0 and record.output_tokens == 0
    verdicts = {
        entry.candidate_id: entry.verdict for entry in record.ineligible
    }
    assert verdicts == {
        wired.candidate_ids[0]: UNRESOLVED,
        wired.candidate_ids[1]: PriorArtVerdict.OVERLAPPING,
    }
    assert wired.store.get_run(record.id) == record


def test_the_no_eligible_record_is_deterministic(tmp_path: Path) -> None:
    first = _Wired(tmp_path / "a", [], (UNRESOLVED,))
    second = _Wired(tmp_path / "b", [], (UNRESOLVED,))
    one = first.selector.run(first.directive).run_record
    two = second.selector.run(second.directive).run_record
    # Run ids are occurrences; everything else is content and must agree.
    def normalize(record: SelectionRunRecord) -> SelectionRunRecord:
        return dataclasses.replace(record, run_id="sel_x", id="")

    assert normalize(one) == normalize(two)


def test_eligibility_reads_the_named_run_only(tmp_path: Path) -> None:
    """A DISTINGUISHED verdict in a later challenge run does not leak
    into a selection whose directive names the earlier run."""
    wired = _Wired(tmp_path, [], (UNRESOLVED,))
    candidate_id = wired.candidate_ids[0]
    later = wired.prior_art_store.record_prior_art_assessment(
        _assessment(candidate_id, DISTINGUISHED, run_id="pac_later")
    )
    wired.prior_art_store.record_run(
        PriorArtRunRecord(
            run_id="pac_later",
            directive_id="pdir_1",
            ideation_run_record_id=wired.record.ideation_run_record_id,
            ideation_run_id="idg_1",
            assessment_id="madq_1",
            map_run_id="map_1",
            snapshot_id="cfp_1",
            candidate_ids=(candidate_id,),
            prior_art_assessment_ids=(later.id,),
            query_execution_ids=(),
            screening_ids=(),
            comparison_ids=(),
            model_calls=3,
            input_tokens=300,
            output_tokens=150,
        )
    )
    result = wired.selector.run(wired.directive)
    assert result.run_record.outcome is (
        SelectionOutcome.NO_ELIGIBLE_CANDIDATE
    )
    assert wired.provider.calls == ()


def test_a_single_eligible_candidate_reviews_with_zero_pairs(
    tmp_path: Path,
) -> None:
    wired, result = _run_happy(tmp_path, (DISTINGUISHED, UNRESOLVED))
    record = result.run_record
    assert record.outcome is SelectionOutcome.SELECTED
    assert record.pairwise_comparisons == ()
    assert len(record.reviews) == 1
    assert record.decision is not None
    assert record.decision.why_selected_over == ()
    assert len(wired.provider.calls) == 2


def test_every_eligible_candidate_disqualified_is_an_honest_stop(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED, DISTINGUISHED))
    wired.provider = FakeModelProvider(
        (
            _review_reply(
                wired.candidate_ids, disqualify=wired.candidate_ids
            ),
        )
    )
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    result = wired.selector.run(wired.directive)
    record = result.run_record
    assert record.outcome is SelectionOutcome.NO_DEFENSIBLE_CANDIDATE
    assert len(wired.provider.calls) == 1
    assert record.decision is None
    assert set(record.disqualified_candidate_ids) == set(
        wired.candidate_ids
    )
    for review in record.reviews:
        assert review.disqualifiers
    assert wired.store.get_run(record.id) == record


# -- bounded repair and accounting -------------------------------------------------


def test_a_schema_violation_buys_one_corrective_call_per_stage(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 2)
    review, decision = (
        _review_reply(wired.candidate_ids),
        _decision_reply(wired.candidate_ids[0], [wired.candidate_ids[1]]),
    )
    wired.provider = FakeModelProvider(
        ("not json", review, "still not json", decision)
    )
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    result = wired.selector.run(wired.directive)
    record = result.run_record
    assert record.outcome is SelectionOutcome.SELECTED
    assert record.model_calls == 4
    assert record.review_provenance is not None
    assert record.review_provenance.repair_count == 1
    assert record.decision is not None
    assert record.decision.provenance.repair_count == 1
    stages = [entry["stage"] for entry in wired.store.rejected()]
    assert sorted(str(stage) for stage in stages) == [
        "comparative_review",
        "selection_decision",
    ]


def test_a_second_schema_violation_on_a_stage_is_final(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 2)
    wired.provider = FakeModelProvider(("not json", "still not json"))
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    with pytest.raises(StructuredOutputError):
        wired.selector.run(wired.directive)
    assert wired.store.runs() == ()
    repairs = sorted(
        int(str(entry["repair"])) for entry in wired.store.rejected()
    )
    assert repairs == [0, 1]


def test_a_billed_schema_violation_stays_on_the_run_record(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 2)
    billed = StructuredOutputError(
        "the reply was truncated mid-object",
        schema=COMPARATIVE_REVIEW_SCHEMA.name,
    ).with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=321, output_tokens=17, model="model-x"
            ),
            latency_seconds=0.5,
        )
    )
    review, decision = (
        _review_reply(wired.candidate_ids),
        _decision_reply(wired.candidate_ids[0], [wired.candidate_ids[1]]),
    )
    wired.provider = FakeModelProvider(
        (ScriptedReply(error=billed), review, decision)
    )
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    result = wired.selector.run(wired.directive)
    record = result.run_record
    drained = wired.ledger.drain()
    assert record.input_tokens == drained.input_tokens
    assert record.output_tokens == drained.output_tokens
    assert record.input_tokens >= 321
    assert record.output_tokens >= 17
    assert record.model_calls == 3


def test_provider_failure_accounts_once_and_accepts_nothing(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 2)
    throttled = ProviderRateLimitError("throttled").with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=50, output_tokens=0, model="model-x"
            ),
            latency_seconds=0.5,
        )
    )
    wired.provider = FakeModelProvider((ScriptedReply(error=throttled),))
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    with pytest.raises(ProviderRateLimitError):
        wired.selector.run(wired.directive)
    drained = wired.ledger.drain()
    assert drained.input_tokens == 50
    assert wired.ledger.drain() == NO_USAGE
    assert wired.store.runs() == ()
    assert wired.store.rejected() == ()


def test_spend_reconciles_exactly_with_the_ledger(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path, (DISTINGUISHED,) * 3)
    record = result.run_record
    drained = wired.ledger.drain()
    assert record.model_calls == drained.calls == 2
    assert record.input_tokens == drained.input_tokens
    assert record.output_tokens == drained.output_tokens


# -- trusted stamps -----------------------------------------------------------------


def test_model_output_cannot_alter_the_stamped_sets(tmp_path: Path) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 3)
    shrunken = _review_reply(wired.candidate_ids[:2])
    wired.provider = FakeModelProvider((shrunken, shrunken))
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    with pytest.raises(SelectionRejectedError, match="missing_decision"):
        wired.selector.run(wired.directive)
    assert wired.store.runs() == ()
    assert len(wired.store.rejected()) == 2


def test_the_stamped_sets_come_from_the_partition(tmp_path: Path) -> None:
    wired, result = _run_happy(
        tmp_path, (DISTINGUISHED, UNRESOLVED, DISTINGUISHED)
    )
    record = result.run_record
    assert record.eligible_candidate_ids == (
        wired.candidate_ids[0],
        wired.candidate_ids[2],
    )
    assert record.ineligible[0].candidate_id == wired.candidate_ids[1]


def test_the_upstream_records_are_untouched(tmp_path: Path) -> None:
    wired = _Wired(tmp_path, [], (DISTINGUISHED,) * 3)
    before_ideas = _digests(tmp_path / "ideation")
    before_prior = _digests(tmp_path / "priorart")
    wired.provider = FakeModelProvider(_happy(wired))
    wired.selector = CandidateSelector(
        provider=wired.provider,
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        store=wired.store,
    )
    wired.selector.run(wired.directive)
    assert _digests(tmp_path / "ideation") == before_ideas
    assert _digests(tmp_path / "priorart") == before_prior


def test_novelty_status_stays_unassessed(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path, (DISTINGUISHED,) * 2)
    assert result.run_record.outcome is SelectionOutcome.SELECTED
    for candidate_id in wired.candidate_ids:
        candidate = wired.ideation_store.get_idea(candidate_id)
        assert candidate is not None
        assert candidate.novelty_status is NoveltyStatus.UNASSESSED


def test_the_run_reloads_intact_from_a_fresh_store(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path, (DISTINGUISHED,) * 3)
    fresh = SelectionStore(tmp_path / "selection")
    assert fresh.get_run(result.run_record.id) == result.run_record
    assert fresh.get_directive(wired.directive.id) == wired.directive


# -- structural score-freedom and the prompt contract --------------------------------


def test_no_schema_carries_a_numeric_field() -> None:
    types_seen: set[str] = set()
    property_names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "type" and isinstance(value, str):
                    types_seen.add(value)
                if key == "properties" and isinstance(value, Mapping):
                    property_names.update(str(name) for name in value)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(COMPARATIVE_REVIEW_SCHEMA.json_schema)
    walk(SELECTION_DECISION_SCHEMA.json_schema)
    assert types_seen.isdisjoint({"number", "integer"})
    banned = ("score", "rank", "rating", "weight")
    for name in property_names:
        assert not any(term in name.lower() for term in banned), name


def test_the_review_instruction_carries_the_directive_constraints(
    tmp_path: Path,
) -> None:
    wired, _ = _run_happy(tmp_path, (DISTINGUISHED,) * 2)
    request = wired.provider.calls[0]
    assert request.instruction == COMPARATIVE_REVIEW_INSTRUCTION
    content = request.messages[0].content
    assert wired.directive.compute_constraint in content
    assert wired.directive.data_constraint in content
    assert wired.directive.time_constraint in content
    assert wired.directive.experimental_constraint in content
    assert "mechanistic accounts of in-context learning" in content


def test_requests_are_deterministic_across_runs(tmp_path: Path) -> None:
    first, _ = _run_happy(tmp_path / "a", (DISTINGUISHED,) * 2)
    second, _ = _run_happy(tmp_path / "b", (DISTINGUISHED,) * 2)
    assert (
        first.provider.calls[0].fingerprint
        == second.provider.calls[0].fingerprint
    )
    assert (
        first.provider.calls[1].fingerprint
        == second.provider.calls[1].fingerprint
    )
