"""The prior-art challenger end to end: deterministic provider and
literature fakes, trusted execution, bounded repair, exact accounting,
and candidate records that stay untouched."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

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
from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureProviderError,
    LiteratureSource,
    ResultOrdering,
    RetrievedSearch,
    ScriptedLiteratureProvider,
)
from autonomous_research_lab.literature.store import LiteratureStore
from autonomous_research_lab.mapping.adequacy import SupportTier
from autonomous_research_lab.mapping.records import (
    CallProvenance,
    ProblemKind,
    ThemeEra,
)
from autonomous_research_lab.priorart.assessment import (
    MissingCandidatePortfolioError,
    PriorArtReasonCode,
    PriorArtThresholds,
    PriorArtVerdict,
)
from autonomous_research_lab.priorart.challenger import (
    PRIOR_ART_QUERY_SCHEMA,
    QUERY_INSTRUCTION,
    PriorArtChallenger,
    PriorArtRejectedError,
)
from autonomous_research_lab.priorart.directive import PriorArtDirective
from autonomous_research_lab.priorart.plan import (
    MAX_ALTERNATIVES_PER_GROUP,
    MAX_CONCEPT_GROUPS,
)
from autonomous_research_lab.priorart.preflight import (
    PriorArtPreflightError,
)
from autonomous_research_lab.priorart.records import (
    DIMENSIONS,
    PriorArtQueryFamily,
)
from autonomous_research_lab.priorart.store import PriorArtStore
from autonomous_research_lab.runtime.metrics import ProviderUsage
from autonomous_research_lab.runtime.providers import (
    CallAccounting,
    FakeModelProvider,
    ProviderTransportError,
    ScriptedReply,
    StructuredOutputError,
    UsageLedger,
)


def _rules_of(rejected: Mapping[str, object]) -> set[str]:
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    return {str(entry["rule"]) for entry in reasons}


CUTOFF = "2026-08-18"
WINDOW = "2025-08-18"

ABSTRACT_CITED = (
    "We show that attention heads specialize during in-context learning "
    "and that ablating them removes the ability."
)
ABSTRACT_A = (
    "We prune attention heads after training and evaluate the pruned "
    "model on language benchmarks."
)
ABSTRACT_B = "We survey optimizers for large-scale vision training."


def _source(
    provider_id: str, title: str, abstract: str | None, **overrides: object
) -> LiteratureSource:
    defaults: dict[str, object] = {
        "provider": "scripted",
        "provider_id": provider_id,
        "title": title,
        "authors": ("Ada Lovelace",),
        "publication_date": "2026-01-15",
        "publication_year": 2026,
        "venue": "Journal of Examples",
        "work_type": "article",
        "abstract": abstract,
        "doi": None,
        "arxiv_id": None,
        "provider_url": f"https://example.org/{provider_id}",
        "landing_page_url": None,
        "pdf_url": None,
        "cited_by_count": 10,
        "referenced_work_ids": (),
        "access_level": (
            AccessLevel.ABSTRACT
            if abstract is not None
            else AccessLevel.METADATA
        ),
    }
    defaults.update(overrides)
    return LiteratureSource(**defaults)  # type: ignore[arg-type]


CITED = _source("W_cited", "Attention Head Specialization", ABSTRACT_CITED)
SRC_A = _source("W_a", "Pruning Attention Heads", ABSTRACT_A)
SRC_B = _source("W_b", "A Survey of Optimizers", ABSTRACT_B)


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_0",
        response_id="mcall_0",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=10,
        output_tokens=10,
        repair_count=0,
    )


def _candidate(cited_ids: tuple[str, ...]) -> CandidateIdea:
    statement = "head-level mechanisms of in-context learning are untested"
    return CandidateIdea(
        run_id="idg_1",
        title="Do Reweighting Scalars Select Induction Heads?",
        research_question="Does head reweighting select induction heads?",
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
        cited_source_ids=cited_ids,
        cited_recent=len(cited_ids),
        cited_foundational=0,
        cited_undated=0,
        provenance=_provenance(),
    )


def _portfolio(candidates: int) -> PortfolioReport:
    return PortfolioReport(
        problems_total=1,
        problems_addressed=1 if candidates else 0,
        problems_unaddressed=0 if candidates else 1,
        unaddressed_statements=()
        if candidates
        else ("an unaddressed problem",),
        addressed_multi_source=0,
        addressed_tentative=1 if candidates else 0,
        addressed_single_source_limitation=0,
        addressed_contradicted=0,
        candidates=candidates,
        distinct_sources_cited=candidates,
        themes_targeted=candidates,
        distinct_problem_sets=candidates,
        distinct_theme_sets=candidates,
        distinct_dataset_sets=candidates,
        distinct_metric_sets=candidates,
    )


def _ideation_run(candidate_ids: tuple[str, ...]) -> IdeationRunRecord:
    return IdeationRunRecord(
        run_id="idg_1",
        directive_id="idir_1",
        assessment_id="madq_1",
        map_run_id="map_1",
        snapshot_id="cfp_1",
        direction_id="dir_1",
        candidate_ids=candidate_ids,
        refusal_justification=""
        if candidate_ids
        else "the problems are saturated",
        diversity_rationale="one candidate" if candidate_ids else "",
        model_calls=2,
        input_tokens=100,
        output_tokens=50,
        portfolio=_portfolio(len(candidate_ids)),
    )


def _retrieved(
    *sources: LiteratureSource, truncated: bool = False
) -> RetrievedSearch:
    return RetrievedSearch(
        provider="scripted",
        retrieved_at="2026-08-19T12:00:00+00:00",
        request_params={"filter": "scripted"},
        total_count=len(sources),
        pages_fetched=1,
        page_identifiers=("",),
        rate_limit={},
        truncated=truncated,
        sources=sources,
    )


#: One retrieval per family, in proposal order: the mechanism search
#: finds the two fresh works, the rest come back empty.
RETRIEVALS: tuple[RetrievedSearch, ...] = (
    _retrieved(SRC_A, SRC_B),
    _retrieved(),
    _retrieved(),
    _retrieved(),
    _retrieved(),
    _retrieved(),
)

#: One distinct, candidate-anchored plan per family, so every
#: rendered query fingerprints distinctly.
PLAN_GROUPS: dict[str, list[list[str]]] = {
    "mechanism": [["attention head reweighting", "head gating"]],
    "problem_mechanism": [
        ["in-context learning"],
        ["attention head reweighting"],
    ],
    "evaluation_setup": [
        ["held-out probes", "probe tasks"],
        ["induction heads"],
    ],
    "synonyms_legacy": [
        ["head gating", "head masking", "head pruning"],
        ["induction heads"],
    ],
    "competing_approaches": [
        ["LoRA", "adapters"],
        ["attention head reweighting"],
    ],
    "recent": [["attention head reweighting"]],
}


def _query_reply(families: tuple[str, ...] | None = None) -> str:
    chosen = families or tuple(f.value for f in PriorArtQueryFamily)
    return json.dumps(
        {
            "queries": [
                {
                    "family": family,
                    "groups": [
                        {"alternatives": list(group)}
                        for group in PLAN_GROUPS[family]
                    ],
                }
                for family in chosen
            ]
        }
    )


QUERY_REPLY = _query_reply()


def _screen_entry(source_id: str, decision: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "decision": decision,
        "reason": "judged from the shown title and abstract",
    }


SCREENING_REPLY = json.dumps(
    {
        "screens": [
            _screen_entry(SRC_A.id, "potential_overlap"),
            _screen_entry(SRC_B.id, "unrelated"),
            _screen_entry(CITED.id, "related"),
        ]
    }
)


def _comparison_entry(
    source_id: str, snippet: str, similarity: str = "related"
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "similarity": similarity,
        "overlap_features": ["both intervene on attention heads"],
        "material_differences": ["reweighting is not pruning or ablation"],
        "dimensions": [
            {
                "dimension": dimension.value,
                "candidate_position": "reweights attention heads",
                "prior_work_position": "operates on attention heads",
                "support_location": "abstract",
                "support_snippet": snippet,
            }
            for dimension in DIMENSIONS
        ],
    }


COMPARISON_REPLY = json.dumps(
    {
        "comparisons": [
            _comparison_entry(SRC_A.id, "prune attention heads"),
            _comparison_entry(CITED.id, "attention heads specialize"),
        ]
    }
)

HAPPY_REPLIES: tuple[str, ...] = (
    QUERY_REPLY,
    SCREENING_REPLY,
    COMPARISON_REPLY,
)

#: Small fixture pool: three unique works instead of ten.
THRESHOLDS = PriorArtThresholds(min_unique_sources=3)


def _directive(**overrides: object) -> PriorArtDirective:
    defaults: dict[str, object] = {
        "ideation_run_record_id": "",
        "cutoff_date": CUTOFF,
        "recent_window_start": WINDOW,
        "results_per_query": 5,
    }
    defaults.update(overrides)
    return PriorArtDirective(**defaults)  # type: ignore[arg-type]


def _challenger(
    tmp_path: Path,
    replies: tuple[ScriptedReply | str, ...],
    *,
    literature_outcomes: tuple[
        RetrievedSearch | LiteratureProviderError, ...
    ] = RETRIEVALS,
    candidates: int = 1,
) -> tuple[
    PriorArtChallenger,
    FakeModelProvider,
    UsageLedger,
    PriorArtStore,
    ScriptedLiteratureProvider,
    str,
]:
    """Full wiring around fakes, plus the ideation-run record id the
    directive must name."""
    ideation_store = IdeationStore(tmp_path / "ideation")
    known = LiteratureStore(tmp_path / "known_literature")
    known.record_source(CITED)
    ids = []
    for _ in range(candidates):
        idea = ideation_store.record_idea(_candidate((CITED.id,)))
        ids.append(idea.id)
    run_record = ideation_store.record_run(_ideation_run(tuple(ids)))
    provider = FakeModelProvider(replies)
    ledger = UsageLedger()
    scripted = ScriptedLiteratureProvider(literature_outcomes)
    store = PriorArtStore(tmp_path / "priorart")
    challenger = PriorArtChallenger(
        provider=provider,
        model="model-x",
        ledger=ledger,
        ideation_store=ideation_store,
        known_literature=known,
        corpus=LiteratureCorpus(
            LiteratureStore(tmp_path / "fresh_literature"), scripted
        ),
        store=store,
        thresholds=THRESHOLDS,
    )
    return challenger, provider, ledger, store, scripted, run_record.id


def test_a_full_challenge_produces_grounded_durable_records(
    tmp_path: Path,
) -> None:
    challenger, provider, ledger, _, _, record_id = _challenger(
        tmp_path, HAPPY_REPLIES
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))

    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()
    assert assessment.compared_work_ids == (SRC_A.id, CITED.id)
    coverage = assessment.coverage
    assert coverage.unique_sources == 3
    assert coverage.known_prior_art_listed == 1
    assert coverage.potential_overlap == 1
    assert coverage.compared_works == 2

    # Full lineage on the run record.
    record = result.run_record
    assert record.ideation_run_record_id == record_id
    assert record.assessment_id == "madq_1"
    assert record.map_run_id == "map_1"
    assert record.snapshot_id == "cfp_1"
    assert record.model_calls == 3
    assert len(record.query_execution_ids) == 6
    assert len(record.screening_ids) == 3
    assert len(record.comparison_ids) == 2

    # The cited source is stamped known prior art by trusted code.
    cited_screen = next(
        s for s in result.screenings if s.source_id == CITED.id
    )
    assert cited_screen.known_prior_art
    fresh_screen = next(
        s for s in result.screenings if s.source_id == SRC_A.id
    )
    assert not fresh_screen.known_prior_art

    # Everything reloads identically through a fresh store.
    fresh = PriorArtStore(tmp_path / "priorart")
    assert fresh.get_run(record.id) == record
    assert fresh.get_prior_art_assessment(assessment.id) == assessment
    assert fresh.assessment_for_candidate(
        record.run_id, result.candidates[0].id
    ) == assessment
    assert ledger.drain().calls == 3
    assert len(provider.calls) == 3


def test_the_guard_runs_before_any_model_call(tmp_path: Path) -> None:
    challenger, provider, _, store, scripted, _ = _challenger(
        tmp_path, HAPPY_REPLIES
    )
    with pytest.raises(MissingCandidatePortfolioError):
        challenger.run(_directive(ideation_run_record_id="irun_missing"))
    assert provider.calls == ()
    assert scripted.queries == ()
    assert store.prior_art_assessments() == ()
    assert store.runs() == ()


def test_trusted_code_sets_every_date_and_ordering(tmp_path: Path) -> None:
    challenger, _, _, _, scripted, record_id = _challenger(
        tmp_path, HAPPY_REPLIES
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    by_family = {e.family: e for e in result.executions}
    assert set(by_family) == set(PriorArtQueryFamily)
    for family, execution in by_family.items():
        assert execution.to_date == CUTOFF
        if family is PriorArtQueryFamily.RECENT:
            assert execution.from_date == WINDOW
        else:
            assert execution.from_date == ""
    influence = {
        PriorArtQueryFamily.MECHANISM,
        PriorArtQueryFamily.SYNONYMS_LEGACY,
        PriorArtQueryFamily.COMPETING_APPROACHES,
    }
    for family, execution in by_family.items():
        expected = (
            ResultOrdering.INFLUENCE
            if family in influence
            else ResultOrdering.RECENCY
        )
        assert execution.ordering is expected
    # The literature layer saw exactly the trusted parameters.
    for query in scripted.queries:
        assert query.to_date == CUTOFF
        assert query.max_results == 5


def test_post_cutoff_sources_are_excluded_before_screening(
    tmp_path: Path,
) -> None:
    late = _source(
        "W_late",
        "A Later Work",
        "We reweight attention heads later.",
        publication_date="2026-08-19",
    )
    retrievals = (_retrieved(SRC_A, SRC_B, late), *RETRIEVALS[1:])
    challenger, _, _, _, _, record_id = _challenger(
        tmp_path, HAPPY_REPLIES, literature_outcomes=retrievals
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    (assessment,) = result.assessments
    assert assessment.coverage.post_cutoff_excluded == 1
    assert assessment.coverage.unique_sources == 4
    screened_ids = {record.source_id for record in result.screenings}
    assert late.id not in screened_ids


SHADOW = _source("W_shadow", "Head Reweighting at Scale", None)

_SHADOW_HYPOTHESIS: dict[str, str] = {
    "candidate_claim": "a causal test of head reweighting",
    "source_text": "head reweighting at scale",
    "support_location": "title",
    "dimension": "mechanism",
    "rationale": (
        "the title names the same head reweighting intervention the "
        "candidate proposes as its contribution"
    ),
}


def test_metadata_only_sources_are_never_compared(tmp_path: Path) -> None:
    # The shadow source screens in its own gated metadata call; its
    # attested potential overlap blocks differentiation but is never
    # rendered for comparison.
    retrievals = (_retrieved(SRC_A, SRC_B, SHADOW), *RETRIEVALS[1:])
    metadata_reply = json.dumps(
        {
            "screens": [
                {
                    "source_id": SHADOW.id,
                    "decision": "potential_overlap",
                    "reason": (
                        "the title names the head reweighting intervention"
                    ),
                    "overlap_hypothesis": dict(_SHADOW_HYPOTHESIS),
                }
            ]
        }
    )
    challenger, provider, _, _, _, record_id = _challenger(
        tmp_path,
        (QUERY_REPLY, SCREENING_REPLY, metadata_reply, COMPARISON_REPLY),
        literature_outcomes=retrievals,
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    (assessment,) = result.assessments
    compared = {record.source_id for record in result.comparisons}
    assert SHADOW.id not in compared
    assert assessment.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
    assert PriorArtReasonCode.METADATA_AMBIGUITY in {
        reason.code for reason in assessment.reasons
    }
    detail = next(
        reason.detail
        for reason in assessment.reasons
        if reason.code is PriorArtReasonCode.METADATA_AMBIGUITY
    )
    assert "a causal test of head reweighting" in detail
    shadow_screen = next(
        record
        for record in result.screenings
        if record.source_id == SHADOW.id
    )
    assert shadow_screen.overlap_hypothesis is not None
    assert (
        shadow_screen.overlap_hypothesis.source_text
        == "head reweighting at scale"
    )
    stages = [str(call.metadata.get("stage")) for call in provider.calls]
    assert stages == [
        "queries",
        "screening",
        "metadata_screening",
        "comparison",
    ]


def test_an_undecidable_metadata_screen_does_not_block(
    tmp_path: Path,
) -> None:
    # The Task 5D.1 failure shape, corrected: a metadata-only source
    # honestly screened undecidable is coverage, and differentiation
    # stays reachable beside it.
    retrievals = (_retrieved(SRC_A, SRC_B, SHADOW), *RETRIEVALS[1:])
    metadata_reply = json.dumps(
        {
            "screens": [
                {
                    "source_id": SHADOW.id,
                    "decision": "undecidable",
                    "reason": "a title alone cannot settle the question",
                }
            ]
        }
    )
    challenger, _, _, _, _, record_id = _challenger(
        tmp_path,
        (QUERY_REPLY, SCREENING_REPLY, metadata_reply, COMPARISON_REPLY),
        literature_outcomes=retrievals,
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
    assert assessment.reasons == ()
    coverage = assessment.coverage
    assert coverage.metadata_level == 1
    assert coverage.undecidable == 1
    assert coverage.metadata_ambiguous == 0


def test_cited_works_screen_before_fresh_ones(tmp_path: Path) -> None:
    # The Task 5D.1 live defect, corrected: cited works sorted last and
    # were exactly the sources the screening cap truncated. They now
    # head the pool — and a directive whose own retrieval could
    # truncate at all is refused by the preflight before any call, so
    # the ordering is defense in depth, not the only guard.
    challenger, provider, _, _, _, record_id = _challenger(
        tmp_path, HAPPY_REPLIES
    )
    challenger.run(_directive(ideation_run_record_id=record_id))
    screening_request = provider.calls[1]
    assert str(screening_request.metadata.get("stage")) == "screening"
    (message,) = screening_request.messages
    assert message.content.index(CITED.id) < message.content.index(SRC_A.id)
    with pytest.raises(PriorArtPreflightError, match="mechanically"):
        challenger.run(
            _directive(
                ideation_run_record_id=record_id,
                max_screened_per_candidate=3,
            )
        )


def test_cited_prior_art_joins_the_pool_deduplicated(
    tmp_path: Path,
) -> None:
    # The fresh search surfaces the SAME work the candidate cites, as a
    # new snapshot (same provider id, different citation count).
    fresh_cited = _source(
        "W_cited",
        "Attention Head Specialization",
        ABSTRACT_CITED,
        cited_by_count=99,
    )
    assert fresh_cited.id != CITED.id
    retrievals = (_retrieved(SRC_A, fresh_cited), *RETRIEVALS[1:])
    screening_reply = json.dumps(
        {
            "screens": [
                _screen_entry(SRC_A.id, "potential_overlap"),
                _screen_entry(fresh_cited.id, "related"),
            ]
        }
    )
    comparison_reply = json.dumps(
        {
            "comparisons": [
                _comparison_entry(SRC_A.id, "prune attention heads"),
                _comparison_entry(
                    fresh_cited.id, "attention heads specialize"
                ),
            ]
        }
    )
    challenger, _, _, _, _, record_id = _challenger(
        tmp_path,
        (QUERY_REPLY, screening_reply, comparison_reply),
        literature_outcomes=retrievals,
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    (assessment,) = result.assessments
    # One representative for the two snapshots of the same work.
    assert assessment.coverage.unique_sources == 2
    assert assessment.coverage.known_prior_art_recovered == 1
    recovered = next(
        s for s in result.screenings if s.source_id == fresh_cited.id
    )
    assert recovered.known_prior_art


def test_a_schema_violation_earns_one_bounded_correction(
    tmp_path: Path,
) -> None:
    challenger, provider, _, store, _, record_id = _challenger(
        tmp_path, ("not json", *HAPPY_REPLIES)
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    assert len(provider.calls) == 4
    (rejected,) = store.rejected()
    assert rejected["stage"] == "queries"
    assert rejected["repair"] == 0
    assert _rules_of(rejected) == {"invalid_structured_output"}
    assert result.run_record.model_calls == 4


def test_a_gate_rejection_earns_one_bounded_correction(
    tmp_path: Path,
) -> None:
    missing_family = _query_reply(
        tuple(
            family.value
            for family in PriorArtQueryFamily
            if family is not PriorArtQueryFamily.SYNONYMS_LEGACY
        )
    )
    challenger, provider, _, store, _, record_id = _challenger(
        tmp_path, (missing_family, *HAPPY_REPLIES)
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    assert len(provider.calls) == 4
    (rejected,) = store.rejected()
    assert _rules_of(rejected) == {"missing_family"}
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED


def test_gate_exhaustion_fails_closed_with_everything_preserved(
    tmp_path: Path,
) -> None:
    missing_family = _query_reply(
        tuple(
            family.value
            for family in PriorArtQueryFamily
            if family is not PriorArtQueryFamily.SYNONYMS_LEGACY
        )
    )
    challenger, provider, _, store, _, record_id = _challenger(
        tmp_path, (missing_family, missing_family)
    )
    directive = _directive(ideation_run_record_id=record_id)
    with pytest.raises(PriorArtRejectedError, match="missing_family"):
        challenger.run(directive)
    assert len(provider.calls) == 2
    assert len(store.rejected()) == 2
    assert store.prior_art_assessments() == ()
    assert store.runs() == ()
    # The completed earlier stage is durable beside the preserved failure.
    fresh = PriorArtStore(tmp_path / "priorart")
    assert fresh.get_directive(directive.id) is not None


def test_a_persistent_schema_violation_reraises_the_typed_error(
    tmp_path: Path,
) -> None:
    challenger, provider, _, store, _, record_id = _challenger(
        tmp_path, ("not json", "still not json")
    )
    with pytest.raises(StructuredOutputError):
        challenger.run(_directive(ideation_run_record_id=record_id))
    assert len(provider.calls) == 2
    assert len(store.rejected()) == 2
    assert store.runs() == ()


def test_an_incoherent_budget_is_refused_before_any_call(
    tmp_path: Path,
) -> None:
    # A directive that cannot cover its own worst case never reaches
    # the provider or the network: the refusal names the arithmetic,
    # and only the directive record — an input, not an outcome — is
    # durable.
    challenger, provider, ledger, store, scripted, record_id = _challenger(
        tmp_path, ()
    )
    with pytest.raises(PriorArtPreflightError, match="worst-case calls"):
        challenger.run(
            _directive(ideation_run_record_id=record_id, max_model_calls=1)
        )
    assert provider.calls == ()
    assert scripted.queries == ()
    assert ledger.drain().calls == 0
    assert store.runs() == ()
    assert tuple((store.root / "executions").glob("*.json")) == ()


def test_the_preflight_collects_every_violation(tmp_path: Path) -> None:
    challenger, _, _, _, _, record_id = _challenger(tmp_path, ())
    with pytest.raises(PriorArtPreflightError) as caught:
        challenger.run(
            _directive(
                ideation_run_record_id=record_id,
                max_screened_per_candidate=10,
                max_model_calls=6,
            )
        )
    message = str(caught.value)
    assert "mechanically truncate" in message
    assert "worst-case calls" in message


def test_a_preflighted_run_stays_inside_its_budget(tmp_path: Path) -> None:
    # The preflight reserves the exact worst case — every gated stage
    # burning its corrective call — so a run it admits completes
    # without ever reaching the runtime budget guard.
    replies = (
        "not json",
        QUERY_REPLY,
        "not json",
        SCREENING_REPLY,
        "not json",
        COMPARISON_REPLY,
    )
    challenger, provider, _, store, _, record_id = _challenger(
        tmp_path, replies
    )
    result = challenger.run(
        _directive(ideation_run_record_id=record_id, max_model_calls=12)
    )
    assert len(provider.calls) == 6
    assert result.run_record.model_calls == 6
    assert len(store.rejected()) == 3
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED


def test_a_billed_schema_violation_stays_on_the_run_record(
    tmp_path: Path,
) -> None:
    # The Muse adapter attaches accounting to a schema violation before
    # raising; the run record must fold that spend exactly as the
    # ledger does — a billed corrective loop may not undercount.
    billed = StructuredOutputError(
        "the reply violated the schema", schema="prior_art_queries"
    ).with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=321, output_tokens=17, model="model-x"
            ),
            latency_seconds=0.5,
        )
    )
    challenger, provider, ledger, _, _, record_id = _challenger(
        tmp_path, (ScriptedReply(error=billed), *HAPPY_REPLIES)
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    assert len(provider.calls) == 4
    drained = ledger.drain()
    assert result.run_record.model_calls == 4
    assert result.run_record.input_tokens == drained.input_tokens
    assert result.run_record.output_tokens == drained.output_tokens
    # The failed call's billed tokens are part of both totals.
    assert result.run_record.input_tokens >= 321
    assert result.run_record.output_tokens >= 17


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
    challenger, _, ledger, store, _, record_id = _challenger(
        tmp_path, (ScriptedReply(error=failure),)
    )
    with pytest.raises(ProviderTransportError, match="connection reset"):
        challenger.run(_directive(ideation_run_record_id=record_id))
    drained = ledger.drain()
    assert drained.input_tokens == 321
    assert ledger.drain().input_tokens == 0
    # A failure is not a gate rejection; nothing was accepted.
    assert store.rejected() == ()
    assert store.prior_art_assessments() == ()
    assert store.runs() == ()


def test_cached_replay_reruns_without_literature_network(
    tmp_path: Path,
) -> None:
    challenger, _, _, _, _, record_id = _challenger(
        tmp_path, HAPPY_REPLIES
    )
    first = challenger.run(_directive(ideation_run_record_id=record_id))
    assert all(not e.from_cache for e in first.executions)

    # A second run over the same corpus root: identical queries replay
    # from the store, and the empty scripted provider proves zero
    # network calls.
    empty = ScriptedLiteratureProvider((), name="scripted")
    replay = PriorArtChallenger(
        provider=FakeModelProvider(HAPPY_REPLIES),
        model="model-x",
        ledger=UsageLedger(),
        ideation_store=IdeationStore(tmp_path / "ideation"),
        known_literature=LiteratureStore(tmp_path / "known_literature"),
        corpus=LiteratureCorpus(
            LiteratureStore(tmp_path / "fresh_literature"), empty
        ),
        store=PriorArtStore(tmp_path / "priorart_replay"),
        thresholds=THRESHOLDS,
    )
    second = replay.run(_directive(ideation_run_record_id=record_id))
    assert all(e.from_cache for e in second.executions)
    assert empty.queries == ()
    assert (
        second.assessments[0].verdict is first.assessments[0].verdict
    )


def test_the_candidate_records_are_untouched(tmp_path: Path) -> None:
    challenger, _, _, _, _, record_id = _challenger(
        tmp_path, HAPPY_REPLIES
    )

    def _digest() -> dict[str, str]:
        return {
            str(path.relative_to(tmp_path)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((tmp_path / "ideation").rglob("*.json"))
        }

    before = _digest()
    challenger.run(_directive(ideation_run_record_id=record_id))
    assert _digest() == before


def test_the_query_instruction_states_the_plan_bounds() -> None:
    # The Task 5D.1 live evidence: all three corrective calls were
    # budget_violation rejections for over-cap alternatives — roughly a
    # fifth of the spent budget lost to a cap the gate enforced but the
    # instruction never stated.
    assert f"{MAX_CONCEPT_GROUPS} at most" in QUERY_INSTRUCTION
    assert (
        f"at most {MAX_ALTERNATIVES_PER_GROUP} per group"
        in QUERY_INSTRUCTION
    )
    schema = str(PRIOR_ART_QUERY_SCHEMA.json_schema)
    assert f"At most {MAX_ALTERNATIVES_PER_GROUP} " in schema
    assert f"{MAX_CONCEPT_GROUPS} is the hard cap" in schema


def test_requests_are_deterministic_across_runs(tmp_path: Path) -> None:
    first, first_provider, _, _, _, first_id = _challenger(
        tmp_path / "one", HAPPY_REPLIES
    )
    second, second_provider, _, _, _, second_id = _challenger(
        tmp_path / "two", HAPPY_REPLIES
    )
    assert first_id == second_id  # content-addressed lineage
    first.run(_directive(ideation_run_record_id=first_id))
    second.run(_directive(ideation_run_record_id=second_id))
    assert [c.fingerprint for c in first_provider.calls] == [
        c.fingerprint for c in second_provider.calls
    ]


def test_the_executed_query_is_the_rendered_boolean(
    tmp_path: Path,
) -> None:
    # The model proposed groups; what actually executed is the trusted
    # renderer's Boolean expression, recorded with its plan and version.
    challenger, _, _, _, scripted, record_id = _challenger(
        tmp_path, HAPPY_REPLIES
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    by_family = {e.family.value: e for e in result.executions}
    mechanism = by_family["mechanism"]
    assert mechanism.text == (
        '("attention head reweighting" OR "head gating")'
    )
    assert mechanism.plan_groups == (
        ("attention head reweighting", "head gating"),
    )
    assert mechanism.renderer == "boolean-v1"
    executed = {query.text for query in scripted.queries}
    assert mechanism.text in executed
    conjunctive = by_family["problem_mechanism"]
    assert conjunctive.text == (
        '("attention head reweighting") AND ("in-context learning")'
    )


def test_an_opaque_query_string_is_not_expressible(
    tmp_path: Path,
) -> None:
    # The Task 5D failure shape as a raw string: the schema has no text
    # field, so the reply is a schema violation, not a search.
    opaque = json.dumps(
        {
            "queries": [
                {
                    "family": family.value,
                    "text": "learned attention head reweighting scalars "
                    "semantic induction heads prefix matching copying",
                }
                for family in PriorArtQueryFamily
            ]
        }
    )
    challenger, provider, _, store, _, record_id = _challenger(
        tmp_path, (opaque, *HAPPY_REPLIES)
    )
    result = challenger.run(_directive(ideation_run_record_id=record_id))
    (rejected,) = store.rejected()
    assert rejected["stage"] == "queries"
    assert _rules_of(rejected) == {"invalid_structured_output"}
    assert len(provider.calls) == 4
    (assessment,) = result.assessments
    assert assessment.verdict is PriorArtVerdict.DISTINGUISHED
