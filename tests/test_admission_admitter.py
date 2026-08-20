"""The admitter end to end: deterministic provider fakes, trusted
copies the model cannot alter, bounded repair, exact accounting,
all-or-nothing writes, zero-call replay, and upstream records that stay
byte-identical."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

import pytest

from autonomous_research_lab.admission import (
    OPERATIONALIZATION_INSTRUCTION,
    AdmissionConflictError,
    AdmissionDirective,
    AdmissionIntegrityError,
    AdmissionRunResult,
    AdmissionStore,
    CandidateAdmitter,
    operationalization_schema,
)
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.prediction import Comparator
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
from autonomous_research_lab.persistence.state_store import SnapshotError
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
    CandidateReview,
    PairwiseComparison,
    SelectionDecision,
    SelectionOutcome,
    SelectionRationale,
    SelectionRunRecord,
)
from autonomous_research_lab.selection.store import SelectionStore

DISTINGUISHED = PriorArtVerdict.DISTINGUISHED


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


def _assessment(candidate_id: str) -> PriorArtAssessment:
    return PriorArtAssessment(
        run_id="pac_1",
        candidate_id=candidate_id,
        directive_id="pdir_1",
        verdict=DISTINGUISHED,
        overlapping_work_ids=(),
        compared_work_ids=("lit_1", "lit_2"),
        reasons=(),
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


class _Wired:
    """One wired admission setup over freshly recorded upstream runs,
    a durable SELECTED selection record included."""

    def __init__(
        self,
        tmp_path: Path,
        replies: Sequence[ScriptedReply | str],
        candidates: int = 2,
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
        titles = ("Alpha", "Beta", "Gamma")
        self.candidates = tuple(
            self.ideation_store.record_idea(_candidate(titles[index]))
            for index in range(candidates)
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
                portfolio=_portfolio(candidates),
            )
        )
        self.assessments = tuple(
            self.prior_art_store.record_prior_art_assessment(
                _assessment(candidate_id)
            )
            for candidate_id in self.candidate_ids
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
        self.selection_directive = self.selection_store.record_directive(
            SelectionDirective(
                prior_art_run_record_id=self.prior_art_run.id,
                compute_constraint="One CPU workstation.",
                data_constraint="Public datasets only.",
                time_constraint="Runs finish within hours.",
                experimental_constraint="Containerized seeded runs.",
            )
        )
        winner = self.candidate_ids[0]
        self.winner_id = winner
        self.selection_run = self.selection_store.record_run(
            SelectionRunRecord(
                run_id="sel_1",
                directive_id=self.selection_directive.id,
                prior_art_run_record_id=self.prior_art_run.id,
                prior_art_run_id="pac_1",
                ideation_run_record_id=self.ideation_run.id,
                ideation_run_id="idg_1",
                direction_id=self.direction.id,
                candidate_ids=self.candidate_ids,
                prior_art_assessment_ids=tuple(
                    a.id for a in self.assessments
                ),
                eligible_candidate_ids=self.candidate_ids,
                ineligible=(),
                disqualified_candidate_ids=(),
                reviews=tuple(
                    _review(candidate_id)
                    for candidate_id in self.candidate_ids
                ),
                pairwise_comparisons=tuple(
                    PairwiseComparison(
                        first_candidate_id=first,
                        second_candidate_id=second,
                        comparison="a grounded comparison",
                    )
                    for first, second in combinations(
                        sorted(self.candidate_ids), 2
                    )
                ),
                review_provenance=_provenance(),
                outcome=SelectionOutcome.SELECTED,
                decision=SelectionDecision(
                    selected_candidate_id=winner,
                    decisive_tradeoff="sharper falsifier at equal cost",
                    why_selected_over=tuple(
                        SelectionRationale(
                            candidate_id=candidate_id,
                            reason="the rival lost on diagnosticity",
                        )
                        for candidate_id in self.candidate_ids
                        if candidate_id != winner
                    ),
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
        self.ledger = UsageLedger()
        self.store = AdmissionStore(tmp_path / "admission")
        self.admitter = CandidateAdmitter(
            provider=self.provider,
            model="model-x",
            ledger=self.ledger,
            ideation_store=self.ideation_store,
            prior_art_store=self.prior_art_store,
            selection_store=self.selection_store,
            store=self.store,
        )


def _reply() -> str:
    return json.dumps(
        {
            "operational_predictions": [
                {
                    "prediction_text": (
                        "ablating reweighted heads drops accuracy"
                    ),
                    "condition": (
                        "held-out probes over the recorded tasks"
                    ),
                    "base_metric": "accuracy",
                    "expected_higher_arm": "ablating reweighted heads",
                    "expected_lower_arm": "random head subsets",
                    "contrary_observation": (
                        "accuracy unchanged after ablation"
                    ),
                    "support": [
                        {
                            "source": "candidate",
                            "field_path": "predictions[0].text",
                            "quote": "drops accuracy",
                        }
                    ],
                }
            ]
        }
    )


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def _run_happy(tmp_path: Path) -> tuple[_Wired, AdmissionRunResult]:
    wired = _Wired(tmp_path, (_reply(),))
    return wired, wired.admitter.run(wired.directive)


# -- the happy path -----------------------------------------------------------


def test_the_happy_path_admits_in_exactly_one_call(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path)

    assert not result.replayed
    assert result.record.model_calls == 1
    assert len(wired.provider.calls) == 1
    assert result.record.selected_candidate_id == wired.winner_id
    assert result.record.selection_run_record_id == wired.selection_run.id
    assert result.record.snapshot_id == wired.snapshot.id
    assert result.record.map_run_id == "map_1"
    assert result.record.map_assessment_id == "madq_1"
    assert result.record.selected_prior_art_assessment_id == (
        wired.assessments[0].id
    )


def test_the_admitted_state_is_a_bare_linked_seed(tmp_path: Path) -> None:
    _, result = _run_happy(tmp_path)
    state = result.state

    assert state.parent_id is None
    assert state.objective == "reproduce the baseline"
    assert state.budget == ResearchBudget.zero()
    (question,) = state.questions
    (hypothesis,) = state.hypotheses
    assert hypothesis.question_id == question.id
    assert len(state.predictions) == 1
    assert state.predictions[0].hypothesis_id == hypothesis.id
    for collection in (
        state.experiments,
        state.results,
        state.evidence_ids,
        state.prediction_tests,
        state.claims,
        state.evidence_links,
        state.assessments,
        state.attempts,
        state.history,
    ):
        assert collection == ()


def test_every_scientific_copy_is_verbatim(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path)
    candidate = wired.candidates[0]
    (question,) = result.state.questions
    (hypothesis,) = result.state.hypotheses

    assert question.text == candidate.research_question
    assert question.importance == candidate.cfp_alignment
    assert hypothesis.statement == candidate.hypothesis
    assert hypothesis.rationale == candidate.mechanism
    assert result.record.measurements == candidate.metrics
    assert result.record.controls == candidate.ablations
    assert result.record.comparison_targets == candidate.baselines
    assert result.record.evaluation_protocol == (
        candidate.evaluation_protocol
    )


def test_the_neutral_encoding_is_structural(tmp_path: Path) -> None:
    """Comparator and threshold are constants of the encoding; the
    contrary region is exactly 'no positive difference'."""
    _, result = _run_happy(tmp_path)
    (prediction,) = result.state.predictions

    assert prediction.comparator is Comparator.GREATER_THAN
    assert prediction.threshold == 0.0
    assert prediction.tolerance == 0.0
    assert prediction.metric == (
        "difference in accuracy: ablating reweighted heads minus "
        "random head subsets"
    )
    assert prediction.expectation == (
        "ablating reweighted heads drops accuracy"
    )
    assert prediction.check(0.5)
    assert not prediction.check(0.0)
    assert not prediction.check(-0.2)
    assert result.record.prediction_ids == (prediction.id,)
    assert result.record.mechanical_reading == "sign_only"


def test_requirements_split_by_provenance(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path)
    inherited = result.record.inherited_requirements
    operator = result.record.operator_requirements

    quotes = {entry.quote for entry in inherited}
    assert "one GPU-day" in quotes  # candidate resources
    assert "One CPU workstation." in quotes  # selection directive
    assert "dataset download" in quotes  # selection decision
    assert {entry.source.value for entry in operator} == {
        "admission_directive"
    }
    assert {entry.quote for entry in operator} == {
        "Batch-scheduled execution.",
        "Jobs bounded to two days.",
        "Checkpoint and resume required.",
    }
    assert all(
        entry.source.value != "admission_directive" for entry in inherited
    )
    assert {entry.record_id for entry in operator} == {wired.directive.id}


def test_the_instruction_and_context_carry_the_statements(
    tmp_path: Path,
) -> None:
    """Attestation fairness: everything the model must stay within is on
    screen verbatim — the seven statements and the labeled fields."""
    wired, _ = _run_happy(tmp_path)
    (request,) = wired.provider.calls
    content = request.messages[0].content

    assert request.instruction == OPERATIONALIZATION_INSTRUCTION
    for statement in (
        "One CPU workstation.",
        "Public datasets only.",
        "Runs finish within hours.",
        "Containerized seeded runs.",
        "Batch-scheduled execution.",
        "Jobs bounded to two days.",
        "Checkpoint and resume required.",
    ):
        assert statement in content
    assert "predictions[0].text: " in content
    assert "predictions[0].falsifier: " in content


def test_spend_reconciles_exactly_with_the_ledger(tmp_path: Path) -> None:
    wired, result = _run_happy(tmp_path)
    drained = wired.ledger.drain()

    assert result.record.input_tokens == drained.input_tokens
    assert result.record.output_tokens == drained.output_tokens
    assert result.record.model_calls == drained.calls == 1


# -- repair and failure boundaries --------------------------------------------


def test_a_schema_violation_buys_exactly_one_corrective(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, ("not json", _reply()))
    result = wired.admitter.run(wired.directive)

    assert result.record.model_calls == 2
    assert len(wired.provider.calls) == 2
    assert len(wired.store.rejected()) == 1
    drained = wired.ledger.drain()
    assert result.record.input_tokens == drained.input_tokens
    assert result.record.output_tokens == drained.output_tokens


def test_a_second_violation_fails_closed_with_nothing_written(
    tmp_path: Path,
) -> None:
    wired = _Wired(tmp_path, ("not json", "still not json"))
    with pytest.raises(StructuredOutputError):
        wired.admitter.run(wired.directive)

    assert wired.store.records() == ()
    assert list((tmp_path / "admission" / "states").glob("*.json")) == []
    assert len(wired.store.rejected()) == 2
    # The directive alone is durable — the documented residue.
    assert wired.store.get_directive(wired.directive.id) is not None


def test_a_billed_failed_call_stays_on_the_record(tmp_path: Path) -> None:
    billed = StructuredOutputError(
        "the reply was not valid JSON", schema="admission_operationalization"
    ).with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=70, output_tokens=9, model="model-x"
            ),
            latency_seconds=0.5,
        )
    )
    wired = _Wired(tmp_path, (ScriptedReply(error=billed), _reply()))
    result = wired.admitter.run(wired.directive)

    drained = wired.ledger.drain()
    assert result.record.model_calls == 2
    assert result.record.input_tokens == drained.input_tokens
    assert result.record.output_tokens == drained.output_tokens
    assert drained.input_tokens >= 70


def test_a_provider_failure_accounts_once_and_admits_nothing(
    tmp_path: Path,
) -> None:
    failure = ProviderRateLimitError("throttled").with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=55, output_tokens=0, model="model-x"
            ),
            latency_seconds=0.5,
        )
    )
    wired = _Wired(tmp_path, (ScriptedReply(error=failure),))
    with pytest.raises(ProviderRateLimitError):
        wired.admitter.run(wired.directive)

    drained = wired.ledger.drain()
    assert drained.input_tokens == 55
    assert wired.store.records() == ()


def test_a_persist_failure_leaves_no_record_and_no_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired = _Wired(tmp_path, (_reply(),))

    def explode(state: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(wired.store, "persist_state", explode)
    with pytest.raises(OSError, match="disk full"):
        wired.admitter.run(wired.directive)

    assert wired.store.records() == ()
    assert list((tmp_path / "admission" / "states").glob("*.json")) == []


def test_a_crash_between_snapshot_and_record_heals_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orphan direction: the snapshot lands, the record write dies.
    Nothing is admitted, and the re-run completes with ONE fresh gated
    call — never claimed to be free — ending with exactly one record."""
    wired = _Wired(tmp_path, (_reply(),))
    original = AdmissionStore.record_admission

    def explode(self: AdmissionStore, record: object) -> object:
        raise OSError("power loss")

    monkeypatch.setattr(AdmissionStore, "record_admission", explode)
    with pytest.raises(OSError, match="power loss"):
        wired.admitter.run(wired.directive)
    monkeypatch.setattr(AdmissionStore, "record_admission", original)

    # Not admitted: the orphan snapshot exists, but no accessor sees it.
    snapshots = list((tmp_path / "admission" / "states").glob("*.json"))
    assert len(snapshots) == 1
    assert wired.store.records() == ()

    rerun = CandidateAdmitter(
        provider=FakeModelProvider((_reply(),)),
        model="model-x",
        ledger=UsageLedger(),
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        selection_store=wired.selection_store,
        store=wired.store,
    )
    result = rerun.run(wired.directive)

    assert not result.replayed
    assert result.record.model_calls == 1  # one fresh call, spent honestly
    assert len(wired.store.records()) == 1
    # At temperature zero the retry re-derived the identical state, so
    # the orphan deduplicated instead of accumulating.
    assert len(
        list((tmp_path / "admission" / "states").glob("*.json"))
    ) == 1


# -- replay and conflict ------------------------------------------------------


def _forbidden_admitter(wired: _Wired) -> CandidateAdmitter:
    """An admitter whose provider refuses every call: anything it
    completes was completed without the model."""
    return CandidateAdmitter(
        provider=FakeModelProvider(()),
        model="model-x",
        ledger=wired.ledger,
        ideation_store=wired.ideation_store,
        prior_art_store=wired.prior_art_store,
        selection_store=wired.selection_store,
        store=wired.store,
    )


def test_replay_returns_the_stored_result_at_zero_calls(
    tmp_path: Path,
) -> None:
    wired, result = _run_happy(tmp_path)
    wired.ledger.drain()
    before = _digests(tmp_path / "admission")

    replayed = _forbidden_admitter(wired).run(wired.directive)

    assert replayed.replayed
    assert replayed.inputs is None
    assert replayed.record == result.record
    assert replayed.state == result.state
    assert wired.ledger.drain() == NO_USAGE
    assert _digests(tmp_path / "admission") == before


def test_replay_needs_no_upstream_stores(tmp_path: Path) -> None:
    """A completed admission replays from the admission root alone —
    the upstream roots may be unmounted entirely."""
    wired, result = _run_happy(tmp_path)
    replayer = CandidateAdmitter(
        provider=FakeModelProvider(()),
        model="model-x",
        ledger=UsageLedger(),
        ideation_store=IdeationStore(tmp_path / "empty-ideation"),
        prior_art_store=PriorArtStore(tmp_path / "empty-priorart"),
        selection_store=SelectionStore(tmp_path / "empty-selection"),
        store=wired.store,
    )

    replayed = replayer.run(wired.directive)

    assert replayed.replayed
    assert replayed.record == result.record


def test_replay_of_a_corrupted_snapshot_is_loud_not_a_fresh_call(
    tmp_path: Path,
) -> None:
    wired, result = _run_happy(tmp_path)
    path = (
        tmp_path
        / "admission"
        / "states"
        / f"{result.state.id}.json"
    )
    path.write_text(path.read_text().replace("reproduce", "doctored"))

    with pytest.raises(SnapshotError):
        _forbidden_admitter(wired).run(wired.directive)


def test_a_second_directive_cannot_replace_an_admitted_selection(
    tmp_path: Path,
) -> None:
    wired, result = _run_happy(tmp_path)
    different = AdmissionDirective(
        selection_run_record_id=wired.selection_run.id,
        scheduling_requirement="A different scheduler.",
        job_duration_requirement="Jobs bounded to two days.",
        checkpoint_requirement="Checkpoint and resume required.",
    )
    rerun = _forbidden_admitter(wired)

    with pytest.raises(
        AdmissionConflictError, match="never silently replaced"
    ) as caught:
        rerun.run(different)
    assert result.record.id in str(caught.value)


# -- integrity ----------------------------------------------------------------


def test_the_upstream_records_are_untouched(tmp_path: Path) -> None:
    wired = _Wired(tmp_path, (_reply(),))
    before = {
        name: _digests(tmp_path / name)
        for name in ("ideation", "priorart", "selection")
    }
    wired.admitter.run(wired.directive)
    after = {
        name: _digests(tmp_path / name)
        for name in ("ideation", "priorart", "selection")
    }
    assert after == before


def test_novelty_status_stays_unassessed(tmp_path: Path) -> None:
    wired, _result = _run_happy(tmp_path)
    reloaded = wired.ideation_store.get_idea(wired.winner_id)
    assert reloaded is not None
    assert reloaded.novelty_status is NoveltyStatus.UNASSESSED


def test_a_fresh_store_reloads_the_admission_identically(
    tmp_path: Path,
) -> None:
    _wired, result = _run_happy(tmp_path)
    fresh = AdmissionStore(tmp_path / "admission")
    record, state = fresh.get_admitted_state(result.record.id)
    assert record == result.record
    assert state == result.state


def test_the_codec_round_trips_the_templated_metric(tmp_path: Path) -> None:
    """persist -> load -> identical id and content, through the long
    templated metric string and the 0.0 float threshold (the
    int-vs-float drift trap in the snapshot codec)."""
    _wired, result = _run_happy(tmp_path)
    fresh = AdmissionStore(tmp_path / "admission")
    _, state = fresh.get_admitted_state(result.record.id)
    assert state.id == result.state.id
    (prediction,) = state.predictions
    assert prediction.threshold == 0.0
    assert isinstance(prediction.threshold, float)
    assert prediction.metric == result.state.predictions[0].metric


def test_the_schema_admits_no_numeric_field(tmp_path: Path) -> None:
    banned_types = {"number", "integer"}
    banned_names = {"score", "rank", "rating", "weight", "confidence"}

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            node_type = node.get("type")
            assert node_type not in banned_types
            properties = node.get("properties")
            if isinstance(properties, Mapping):
                for name in properties:
                    assert not any(
                        banned in str(name).casefold()
                        for banned in banned_names
                    )
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(operationalization_schema(("accuracy",)).json_schema)


def test_a_tampered_admitted_budget_is_loud(tmp_path: Path) -> None:
    """A ResearchState's content id excludes its budget, so a doctored
    budget reloads with a clean id; the admission accessor still refuses
    because an admitted seed's budget is zero by construction."""
    wired, result = _run_happy(tmp_path)
    path = (
        tmp_path / "admission" / "states" / f"{result.state.id}.json"
    )
    payload = json.loads(path.read_text())
    payload["budget"]["usd"] = 5.0
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    with pytest.raises(AdmissionIntegrityError, match="non-zero budget"):
        wired.store.get_admitted_state(result.record.id)
