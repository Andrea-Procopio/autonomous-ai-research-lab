"""The gated idea generator end to end, on the fake provider and a real
mapping store. The invariants under test are the seams: the adequacy
guard before any spend, deterministic input verification, the bounded
corrective call, trusted tier and era stamping, honest refusal and
honest portfolio accounting, exact usage accounting, budget enforcement,
and durable provenance. No test opens a network connection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from autonomous_research_lab.ideation.direction import CfpSnapshot
from autonomous_research_lab.ideation.directive import IdeationDirective
from autonomous_research_lab.ideation.generator import (
    IdeaGenerator,
    IdeationBudgetError,
    IdeationContractError,
    IdeationRejectedError,
)
from autonomous_research_lab.ideation.records import (
    NoveltyStatus,
    problem_key,
    theme_key,
)
from autonomous_research_lab.ideation.store import IdeationStore
from autonomous_research_lab.mapping.adequacy import (
    AdequacyMetrics,
    AdequacyReason,
    AdequacyReasonCode,
    AdequacyStatus,
    AdequacyThresholds,
    InadequateFieldMapError,
    MapAdequacyAssessment,
    ProblemSupport,
    SupportTier,
    require_adequate_for_idea_generation,
    support_tier,
)
from autonomous_research_lab.mapping.brief import ResearchBrief, SourceEra
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    DatasetUse,
    ExtractionRecord,
    FieldMapRecord,
    ProblemEntry,
    ProblemInventoryRecord,
    ProblemKind,
    SupportLocation,
    ThemeEntry,
    ThemeEra,
)
from autonomous_research_lab.mapping.store import MappingStore
from autonomous_research_lab.runtime.metrics import ProviderUsage
from autonomous_research_lab.runtime.providers import (
    CallAccounting,
    FakeModelProvider,
    ProviderTransportError,
    ScriptedReply,
    StructuredOutputError,
    UsageLedger,
)

MAP_RUN = "map_t1"

BRIEF = ResearchBrief(
    topic="in-context learning in large language models",
    cutoff_date="2026-08-18",
    recent_window_start="2025-08-18",
)


def _map_provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_map",
        response_id="mcall_map",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=50,
        output_tokens=80,
        repair_count=0,
    )


def _extraction(
    source_id: str, era: SourceEra, **overrides: object
) -> ExtractionRecord:
    values: dict[str, object] = {
        "run_id": MAP_RUN,
        "source_id": source_id,
        "era": era,
        "access_level": "abstract",
        "support_location": SupportLocation.ABSTRACT,
        "sufficient_support": True,
        "insufficiency_reason": "",
        "methods": (),
        "datasets": (),
        "metrics": (),
        "evaluation_protocols": (),
        "baselines": (),
        "reported_results": (),
        "limitations": (),
        "future_work": (),
        "open_problems": (),
        "provenance": _map_provenance(),
    }
    values.update(overrides)
    return ExtractionRecord(**values)  # type: ignore[arg-type]


EXTRACT_R = _extraction(
    "lit_r",
    SourceEra.RECENT,
    methods=("attention head reweighting",),
    datasets=(DatasetUse(name="GLUE"),),
    metrics=("accuracy",),
    baselines=("LoRA",),
    reported_results=("improves accuracy by 2.4 points",),
    open_problems=("which heads carry in-context learning",),
)
EXTRACT_F = _extraction(
    "lit_f",
    SourceEra.FOUNDATIONAL,
    methods=("kalman filtering account of in-context learning",),
    reported_results=("evaluated on synthetic regression only",),
)

P_OPEN = ProblemEntry(
    statement=(
        "Head-level mechanisms of in-context learning remain unclear."
    ),
    kind=ProblemKind.OPEN_PROBLEM,
    grounding="Both records report open mechanism questions.",
    supporting_source_ids=("lit_r", "lit_f"),
)
P_LIMIT = ProblemEntry(
    statement="The filtering account rests on synthetic regression only.",
    kind=ProblemKind.DATA_LIMITATION,
    grounding="One record reports synthetic-only evaluation.",
    supporting_source_ids=("lit_f",),
)
P_CONFLICT = ProblemEntry(
    statement="Theoretical accounts of in-context learning disagree.",
    kind=ProblemKind.CONFLICTING_FINDINGS,
    grounding="The records give incompatible accounts.",
    supporting_source_ids=("lit_r",),
    conflicting_source_ids=("lit_f",),
)

FIELD_MAP = FieldMapRecord(
    run_id=MAP_RUN,
    brief_id=BRIEF.id,
    themes=(
        ThemeEntry(
            name="Mechanistic accounts",
            summary="How in-context learning works inside the model.",
            era=ThemeEra.BOTH,
            source_ids=("lit_r", "lit_f"),
        ),
    ),
    approaches=(),
    evaluation_practices=(),
    relationships=(),
    recent_source_ids=("lit_r",),
    foundational_source_ids=("lit_f",),
    undated_source_ids=(),
    provenance=_map_provenance(),
)

INVENTORY = ProblemInventoryRecord(
    run_id=MAP_RUN,
    brief_id=BRIEF.id,
    problems=(P_OPEN, P_LIMIT, P_CONFLICT),
    provenance=_map_provenance(),
)


def _metrics() -> AdequacyMetrics:
    return AdequacyMetrics(
        screened=10,
        relevant_sources=8,
        excluded_sources=1,
        uncertain_sources=1,
        uncertain_fraction=0.1,
        grounded_sources=2,
        insufficient_extractions=0,
        metadata_only_relevant=0,
        recent_grounded=1,
        foundational_grounded=1,
        undated_grounded=0,
        families_with_relevant=("foundational", "methods", "recent"),
        total_retrieved=12,
        unique_sources=12,
        overlap=0,
        saturation=0.0,
        screening_truncated=0,
        extraction_truncated=0,
        multi_source_themes=1,
        single_source_themes=0,
        multi_source_problems=1,
        tentative_problems=0,
        single_source_limitation_problems=1,
        contradicted_problems=1,
    )


ASSESSMENT = MapAdequacyAssessment(
    run_id=MAP_RUN,
    brief_id=BRIEF.id,
    field_map_id=FIELD_MAP.id,
    inventory_id=INVENTORY.id,
    status=AdequacyStatus.ADEQUATE_FOR_IDEA_GENERATION,
    reasons=(),
    thresholds=AdequacyThresholds(),
    metrics=_metrics(),
    problem_support=tuple(
        ProblemSupport(
            statement=problem.statement,
            kind=problem.kind,
            tier=support_tier(problem),
            distinct_supporting=len(set(problem.supporting_source_ids)),
            conflicting=len(set(problem.conflicting_source_ids)),
        )
        for problem in INVENTORY.problems
    ),
)

CALL_TEXT = (
    "Workshop on In-Context Learning.\n"
    "Topics of interest:\n"
    "- mechanisms of in-context learning\n"
    "- efficient adaptation\n"
    "Submissions are limited to 9 pages.\n"
    "Papers due 30 September 2026.\n"
)
SNAPSHOT = CfpSnapshot(
    source_url="https://example.org/workshop/cfp",
    supplied_at="2026-08-19T09:00:00",
    text=CALL_TEXT,
)
DIRECTIVE = IdeationDirective(
    assessment_id=ASSESSMENT.id, snapshot_id=SNAPSHOT.id
)

DIRECTION_REPLY = json.dumps(
    {
        "scope": (
            "A call for work on how in-context learning operates and on "
            "efficient adaptation."
        ),
        "topics": [
            "mechanisms of in-context learning",
            "efficient adaptation",
        ],
        "constraints": ["Submissions are limited to 9 pages."],
        "relevant_dates": ["30 September 2026"],
    }
)


def _candidate_json(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "title": "Head reweighting under domain shift",
        "research_question": (
            "Does attention head reweighting keep its gains under "
            "distribution shift?"
        ),
        "proposed_contribution": (
            "An out-of-domain evaluation of head reweighting."
        ),
        "mechanism": (
            "Reweighted heads carry task identity that may be domain "
            "specific."
        ),
        "hypothesis": (
            "Reweighting keeps most of its in-domain gain out of domain."
        ),
        "grounding": (
            "One cited record reports reweighting improving accuracy by "
            "2.4 points on GLUE."
        ),
        "predictions": [
            {
                "text": "Out-of-domain accuracy drops by at most 5 points.",
                "falsifier": (
                    "Out-of-domain accuracy drops by more than 5 points."
                ),
            }
        ],
        "datasets": [
            {
                "name": "GLUE",
                "status": "existing",
                "role": "in-domain evaluation",
            }
        ],
        "metrics": ["accuracy"],
        "evaluation_protocol": (
            "Adapt in one domain, evaluate in a held-out domain."
        ),
        "baselines": ["full fine-tuning"],
        "ablations": ["remove the reweighting scalars"],
        "resources": {
            "compute": "a few GPU days",
            "data": "public benchmarks",
            "implementation": "a small adapter patch",
        },
        "risks": ["the effect may vanish under shift"],
        "cfp_alignment": "Addresses the mechanisms topic of the call.",
        "aligned_topics": ["mechanisms of in-context learning"],
        "uncertainty": (
            "Grounded in abstract-level claims from two records only."
        ),
        "search_terms": ["attention head reweighting robustness"],
        "problem_keys": [problem_key(P_OPEN.statement)],
        "theme_keys": [theme_key("Mechanistic accounts")],
        "cited_source_ids": ["lit_r", "lit_f"],
    }
    values.update(overrides)
    return values


CANDIDATE_A = _candidate_json()
CANDIDATE_B = _candidate_json(
    title="Reconciling filtering and mechanistic accounts",
    research_question=(
        "Which account of in-context learning predicts behaviour on "
        "shared tasks?"
    ),
    mechanism=(
        "The two accounts imply different error patterns on the same "
        "prompts."
    ),
    hypothesis=(
        "The accounts disagree measurably on at least one shared task "
        "family."
    ),
    grounding=(
        "The cited records give incompatible accounts of in-context "
        "learning."
    ),
    predictions=[
        {
            "text": (
                "The accounts rank the same systems differently on "
                "shared prompts."
            ),
            "falsifier": (
                "Both accounts produce identical error patterns on "
                "every shared prompt family."
            ),
        }
    ],
    datasets=[
        {
            "name": "a new shared diagnostic suite",
            "status": "new_requirement",
            "role": "head-to-head comparison",
        }
    ],
    metrics=["rank agreement"],
    problem_keys=[problem_key(P_CONFLICT.statement)],
    aligned_topics=["efficient adaptation"],
    search_terms=["in-context learning theory comparison"],
)


def _candidates_reply(*candidates: dict[str, object]) -> str:
    return json.dumps(
        {
            "candidates": list(candidates),
            "diversity_rationale": (
                "The candidates target distinct problems with distinct "
                "mechanisms and evaluation settings."
            ),
            "refusal_justification": "",
        }
    )


CANDIDATES_REPLY = _candidates_reply(CANDIDATE_A, CANDIDATE_B)
HAPPY_REPLIES: tuple[ScriptedReply | str, ...] = (
    DIRECTION_REPLY,
    CANDIDATES_REPLY,
)


def _rules_of(rejected: Mapping[str, object]) -> set[str]:
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    rules: set[str] = set()
    for reason in reasons:
        assert isinstance(reason, Mapping)
        rules.add(str(reason["rule"]))
    return rules


def _populated_map_store(root: Path) -> MappingStore:
    store = MappingStore(root / "mapping")
    store.record_brief(BRIEF)
    store.record_extraction(EXTRACT_R)
    store.record_extraction(EXTRACT_F)
    store.record_field_map(FIELD_MAP)
    store.record_inventory(INVENTORY)
    store.record_adequacy(ASSESSMENT)
    return store


def _generator(
    root: Path, replies: tuple[ScriptedReply | str, ...]
) -> tuple[
    IdeaGenerator, FakeModelProvider, UsageLedger, IdeationStore, MappingStore
]:
    map_store = _populated_map_store(root)
    provider = FakeModelProvider(replies)
    ledger = UsageLedger()
    store = IdeationStore(root / "ideation")
    generator = IdeaGenerator(
        provider=provider,
        model="model-x",
        ledger=ledger,
        map_store=map_store,
        store=store,
    )
    return generator, provider, ledger, store, map_store


def test_a_full_ideation_run_produces_grounded_durable_records(
    tmp_path: Path,
) -> None:
    generator, provider, ledger, _, map_store = _generator(
        tmp_path, HAPPY_REPLIES
    )
    result = generator.run(DIRECTIVE, SNAPSHOT)

    # Two gated stages, one call each, staged metadata included.
    assert len(provider.calls) == 2
    assert provider.calls[0].metadata["stage"] == "direction"
    assert provider.calls[1].metadata["stage"] == "candidates"
    assert result.direction.topics == (
        "mechanisms of in-context learning",
        "efficient adaptation",
    )

    # Trusted stamping: statements, kinds, tiers, theme eras, source
    # eras, and the structural novelty status — none of them appear in
    # the scripted reply.
    assert "multi_source" not in CANDIDATES_REPLY
    first, second = result.ideas
    (addressed_a,) = first.addressed_problems
    assert addressed_a.statement == P_OPEN.statement
    assert addressed_a.kind is ProblemKind.OPEN_PROBLEM
    assert addressed_a.tier is SupportTier.MULTI_SOURCE
    (addressed_b,) = second.addressed_problems
    assert addressed_b.tier is SupportTier.CONTRADICTED
    (theme,) = first.targeted_themes
    assert theme.name == "Mechanistic accounts"
    assert theme.era is ThemeEra.BOTH
    assert (first.cited_recent, first.cited_foundational) == (1, 1)
    assert first.novelty_status is NoveltyStatus.UNASSESSED
    assert first.provenance.repair_count == 0

    # Honest portfolio accounting: the unaddressed problem is named.
    portfolio = result.run_record.portfolio
    assert portfolio.problems_total == 3
    assert portfolio.problems_addressed == 2
    assert portfolio.unaddressed_statements == (P_LIMIT.statement,)
    assert portfolio.addressed_multi_source == 1
    assert portfolio.addressed_contradicted == 1
    assert portfolio.candidates == 2
    assert portfolio.distinct_problem_sets == 2
    assert portfolio.distinct_dataset_sets == 2

    # Exact accounting: the ledger and the run record agree, once.
    usage = ledger.drain()
    assert usage.calls == 2 == result.run_record.model_calls
    assert usage.input_tokens == result.run_record.input_tokens
    assert usage.output_tokens == result.run_record.output_tokens
    assert ledger.drain().calls == 0

    # Durable reload from a fresh store, rejected empty.
    fresh = IdeationStore(tmp_path / "ideation")
    assert fresh.get_run(result.run_record.id) == result.run_record
    for idea in result.ideas:
        assert fresh.get_idea(idea.id) == idea
    assert fresh.get_directive(DIRECTIVE.id) == DIRECTIVE
    assert fresh.get_snapshot(SNAPSHOT.id) == SNAPSHOT
    assert fresh.get_direction(result.direction.id) == result.direction
    assert fresh.rejected() == ()
    assert fresh.runs_for_assessment(ASSESSMENT.id) == (result.run_record,)

    # The read-only mapping input still walks the guard, unchanged.
    assert (
        require_adequate_for_idea_generation(map_store, ASSESSMENT.id)
        == result.assessment
    )


def test_the_guard_refuses_before_any_model_call(tmp_path: Path) -> None:
    generator, provider, ledger, store, map_store = _generator(
        tmp_path, HAPPY_REPLIES
    )
    insufficient = MapAdequacyAssessment(
        run_id="map_thin",
        brief_id=BRIEF.id,
        field_map_id=FIELD_MAP.id,
        inventory_id=INVENTORY.id,
        status=AdequacyStatus.INSUFFICIENT_COVERAGE,
        reasons=(
            AdequacyReason(
                AdequacyReasonCode.TOO_FEW_RELEVANT,
                "2 relevant source(s) of 10 screened; the bar is 8",
            ),
        ),
        thresholds=AdequacyThresholds(),
        metrics=_metrics(),
        problem_support=(),
    )
    map_store.record_adequacy(insufficient)
    thin_directive = IdeationDirective(
        assessment_id=insufficient.id, snapshot_id=SNAPSHOT.id
    )
    with pytest.raises(InadequateFieldMapError, match="not adequate"):
        generator.run(thin_directive, SNAPSHOT)
    with pytest.raises(InadequateFieldMapError, match="no adequacy"):
        generator.run(
            IdeationDirective(
                assessment_id="madq_missing", snapshot_id=SNAPSHOT.id
            ),
            SNAPSHOT,
        )
    # Zero spend, zero records beyond the durable intent.
    assert provider.calls == ()
    assert ledger.drain().calls == 0
    assert store.get_directive(thin_directive.id) is not None
    assert store.get_snapshot(SNAPSHOT.id) is not None
    assert store.runs() == ()
    assert store.ideas() == ()


def test_a_mismatched_snapshot_is_a_contract_error(tmp_path: Path) -> None:
    generator, provider, _, _, _ = _generator(tmp_path, HAPPY_REPLIES)
    other = CfpSnapshot(
        source_url="https://example.org/another",
        supplied_at="2026-08-19T09:00:00",
        text="A different call.",
    )
    with pytest.raises(IdeationContractError, match="names snapshot"):
        generator.run(DIRECTIVE, other)
    assert provider.calls == ()


def test_an_assessment_from_another_store_is_a_contract_error(
    tmp_path: Path,
) -> None:
    bare_map = MappingStore(tmp_path / "bare")
    bare_map.record_adequacy(ASSESSMENT)
    provider = FakeModelProvider(HAPPY_REPLIES)
    generator = IdeaGenerator(
        provider=provider,
        model="model-x",
        ledger=UsageLedger(),
        map_store=bare_map,
        store=IdeationStore(tmp_path / "ideation"),
    )
    with pytest.raises(IdeationContractError, match="different store"):
        generator.run(DIRECTIVE, SNAPSHOT)
    assert provider.calls == ()


def test_a_schema_violation_earns_one_bounded_correction(
    tmp_path: Path,
) -> None:
    generator, provider, _, store, _ = _generator(
        tmp_path, ("this is not json", DIRECTION_REPLY, CANDIDATES_REPLY)
    )
    result = generator.run(DIRECTIVE, SNAPSHOT)
    assert len(provider.calls) == 3
    assert result.run_record.model_calls == 3
    assert result.direction.provenance.repair_count == 1
    (rejected,) = store.rejected()
    assert rejected["stage"] == "direction"
    assert _rules_of(rejected) == {"invalid_structured_output"}


def test_a_gate_rejection_earns_one_bounded_correction(
    tmp_path: Path,
) -> None:
    shorthand = _candidates_reply(_candidate_json(problem_keys=["P1"]))
    generator, provider, ledger, store, _ = _generator(
        tmp_path, (DIRECTION_REPLY, shorthand, CANDIDATES_REPLY)
    )
    result = generator.run(DIRECTIVE, SNAPSHOT)
    assert len(provider.calls) == 3
    assert ledger.drain().calls == 3
    (rejected,) = store.rejected()
    assert rejected["stage"] == "candidates"
    assert _rules_of(rejected) == {"unknown_problem"}
    for idea in result.ideas:
        assert idea.provenance.repair_count == 1


def test_gate_exhaustion_fails_closed_with_everything_preserved(
    tmp_path: Path,
) -> None:
    shorthand = _candidates_reply(_candidate_json(problem_keys=["P1"]))
    generator, provider, _, store, _ = _generator(
        tmp_path, (DIRECTION_REPLY, shorthand, shorthand)
    )
    with pytest.raises(IdeationRejectedError, match="unknown_problem"):
        generator.run(DIRECTIVE, SNAPSHOT)
    assert len(provider.calls) == 3
    assert len(store.rejected()) == 2
    assert store.ideas() == ()
    assert store.runs() == ()
    # The completed first stage is durable beside the preserved failure.
    assert len(store.runs()) == 0
    fresh = IdeationStore(tmp_path / "ideation")
    assert fresh.get_directive(DIRECTIVE.id) is not None


def test_a_persistent_schema_violation_reraises_the_typed_error(
    tmp_path: Path,
) -> None:
    generator, provider, _, store, _ = _generator(
        tmp_path, ("not json", "still not json")
    )
    with pytest.raises(StructuredOutputError):
        generator.run(DIRECTIVE, SNAPSHOT)
    assert len(provider.calls) == 2
    assert len(store.rejected()) == 2
    assert store.runs() == ()


def test_the_model_call_budget_fails_closed(tmp_path: Path) -> None:
    small = IdeationDirective(
        assessment_id=ASSESSMENT.id,
        snapshot_id=SNAPSHOT.id,
        max_model_calls=1,
    )
    generator, provider, ledger, store, _ = _generator(
        tmp_path, (DIRECTION_REPLY,)
    )
    with pytest.raises(IdeationBudgetError, match="budget"):
        generator.run(small, SNAPSHOT)
    # The direction stage spent the one allowed call; the candidates
    # call was refused before reaching the provider.
    assert len(provider.calls) == 1
    assert ledger.drain().calls == 1
    assert store.runs() == ()
    assert len(tuple((store.root / "directions").glob("*.json"))) == 1


def test_provider_failure_accounts_once_and_accepts_nothing(
    tmp_path: Path,
) -> None:
    failure = ProviderTransportError("connection reset").with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=321, output_tokens=0, model="model-x"
            ),
            latency_seconds=0.5,
        )
    )
    generator, _, ledger, store, _ = _generator(
        tmp_path, (ScriptedReply(error=failure),)
    )
    with pytest.raises(ProviderTransportError, match="connection reset"):
        generator.run(DIRECTIVE, SNAPSHOT)
    drained = ledger.drain()
    assert drained.input_tokens == 321
    assert ledger.drain().input_tokens == 0
    # A failure is not a rejection; the durable intent remains.
    assert store.rejected() == ()
    assert store.get_directive(DIRECTIVE.id) is not None
    assert store.get_snapshot(SNAPSHOT.id) is not None
    assert store.runs() == ()


def test_an_honest_refusal_is_recorded_durably(tmp_path: Path) -> None:
    refusal_reply = json.dumps(
        {
            "candidates": [],
            "diversity_rationale": "",
            "refusal_justification": (
                "The mapped problems rest on two abstract-level records; "
                "none supports a defensible candidate under this call."
            ),
        }
    )
    generator, provider, _, _, _ = _generator(
        tmp_path, (DIRECTION_REPLY, refusal_reply)
    )
    result = generator.run(DIRECTIVE, SNAPSHOT)
    assert result.ideas == ()
    record = result.run_record
    assert record.candidate_ids == ()
    assert record.refusal_justification.startswith("The mapped problems")
    assert record.diversity_rationale == ""
    assert record.portfolio.candidates == 0
    assert record.portfolio.problems_unaddressed == 3
    assert set(record.portfolio.unaddressed_statements) == {
        P_OPEN.statement,
        P_LIMIT.statement,
        P_CONFLICT.statement,
    }
    assert len(provider.calls) == 2
    fresh = IdeationStore(tmp_path / "ideation")
    assert fresh.get_run(record.id) == record
    assert fresh.ideas() == ()


def test_unaddressed_problems_are_reported_honestly(tmp_path: Path) -> None:
    generator, _, _, _, _ = _generator(
        tmp_path, (DIRECTION_REPLY, _candidates_reply(CANDIDATE_A))
    )
    result = generator.run(DIRECTIVE, SNAPSHOT)
    portfolio = result.run_record.portfolio
    assert portfolio.candidates == 1
    assert portfolio.problems_addressed == 1
    assert portfolio.unaddressed_statements == (
        P_LIMIT.statement,
        P_CONFLICT.statement,
    )
    assert portfolio.addressed_multi_source == 1
    assert portfolio.addressed_contradicted == 0


def test_tier_and_era_stamps_come_from_trusted_code(tmp_path: Path) -> None:
    limitation_candidate = _candidate_json(
        title="Beyond synthetic regression for the filtering account",
        research_question=(
            "Does the filtering account survive natural-data evaluation?"
        ),
        mechanism=(
            "The filtering account may depend on properties unique to "
            "synthetic regression."
        ),
        hypothesis=(
            "The account's fit degrades on natural task distributions."
        ),
        grounding=(
            "The cited record reports evaluation on synthetic regression "
            "only."
        ),
        datasets=[
            {
                "name": "a natural-task probe set",
                "status": "new_requirement",
                "role": "evaluation beyond synthetic data",
            }
        ],
        problem_keys=[problem_key(P_LIMIT.statement)],
        cited_source_ids=["lit_f"],
    )
    reply = _candidates_reply(limitation_candidate)
    assert "single_source_limitation" not in reply
    assert "foundational" not in reply
    generator, _, _, _, _ = _generator(
        tmp_path, (DIRECTION_REPLY, reply)
    )
    result = generator.run(DIRECTIVE, SNAPSHOT)
    (idea,) = result.ideas
    (addressed,) = idea.addressed_problems
    assert addressed.tier is SupportTier.SINGLE_SOURCE_LIMITATION
    assert addressed.kind is ProblemKind.DATA_LIMITATION
    assert addressed.statement == P_LIMIT.statement
    assert idea.cited_foundational == 1
    assert idea.cited_recent == 0
    assert (
        result.run_record.portfolio.addressed_single_source_limitation == 1
    )


def test_requests_are_deterministic_across_runs(tmp_path: Path) -> None:
    first_generator, first_provider, _, _, _ = _generator(
        tmp_path / "one", HAPPY_REPLIES
    )
    second_generator, second_provider, _, _, _ = _generator(
        tmp_path / "two", HAPPY_REPLIES
    )
    first_generator.run(DIRECTIVE, SNAPSHOT)
    second_generator.run(DIRECTIVE, SNAPSHOT)
    # Identical inputs build identical requests — the fingerprint
    # excludes the per-run metadata, which differs by occurrence id.
    for index in range(2):
        assert (
            first_provider.calls[index].fingerprint
            == second_provider.calls[index].fingerprint
        )
    assert (
        first_provider.calls[0].metadata["ideation_run"]
        != second_provider.calls[0].metadata["ideation_run"]
    )
