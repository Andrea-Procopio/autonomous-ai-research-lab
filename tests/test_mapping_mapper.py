"""The field mapper end to end, on FakeModelProvider and a static Task 5A
corpus. The invariants under test are the seams: trusted retrieval,
preserved verdicts, deterministic insufficiency, the bounded corrective
call, exact usage accounting, budget enforcement, cached replay, and
durable provenance. No test opens a network connection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.literature.corpus import LiteratureCorpus
from autonomous_research_lab.literature.retrieval import (
    AccessLevel,
    LiteratureProviderError,
    LiteratureSource,
    RetrievedSearch,
    ScriptedLiteratureProvider,
)
from autonomous_research_lab.literature.store import LiteratureStore
from autonomous_research_lab.mapping.brief import ResearchBrief, SourceEra
from autonomous_research_lab.mapping.mapper import (
    FieldMapper,
    MappingBudgetError,
    MappingContractError,
    MappingRejectedError,
)
from autonomous_research_lab.mapping.records import (
    ScreeningDecision,
    SupportLocation,
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

BRIEF = ResearchBrief(
    topic="in-context learning",
    cutoff_date="2026-08-18",
    recent_window_start="2026-01-01",
    workshop_hints=("efficient adaptation of frozen models",),
    max_queries_per_family=1,
    results_per_query=10,
    max_screened_sources=10,
    max_extracted_sources=10,
    max_model_calls=20,
)


def _source(
    provider_id: str,
    *,
    date: str | None,
    abstract: str | None,
    title: str,
) -> LiteratureSource:
    return LiteratureSource(
        provider="scripted",
        provider_id=provider_id,
        title=title,
        authors=("Ada Lovelace",),
        publication_date=date,
        publication_year=int(date[:4]) if date else None,
        venue="Journal of Things",
        work_type="article",
        abstract=abstract,
        doi=None,
        arxiv_id=None,
        provider_url=f"https://example.org/{provider_id}",
        landing_page_url=None,
        pdf_url=None,
        cited_by_count=None,
        referenced_work_ids=(),
        access_level=(
            AccessLevel.ABSTRACT if abstract else AccessLevel.METADATA
        ),
    )


S_RECENT = _source(
    "W-recent",
    date="2026-07-01",
    title="Prompt Adaptation for In-Context Learning",
    abstract=(
        "We study prompt adaptation for in-context learning. On the GLUE "
        "benchmark our method reaches 88.5 accuracy over 3 seeds, but "
        "degrades under distribution shift."
    ),
)
S_META = _source(
    "W-meta",
    date="2026-06-15",
    title="In-Context Learning at the Edge",
    abstract=None,
)
S_UNCERTAIN = _source(
    "W-uncertain",
    date="2026-04-04",
    title="Adaptation in Changing Environments",
    abstract="We discuss adaptation in changing environments.",
)
S_FOUNDATIONAL = _source(
    "W-foundational",
    date="2019-06-01",
    title="Episodic Meta-Learning",
    abstract=(
        "Meta-learning trains models episodically for rapid adaptation; "
        "evaluation uses accuracy on held-out tasks."
    ),
)
S_EXCLUDED = _source(
    "W-excluded",
    date="2026-05-05",
    title="Deep-Sea Coral Reproduction",
    abstract="We survey deep-sea coral reproduction.",
)


def _retrieved(*sources: LiteratureSource) -> RetrievedSearch:
    return RetrievedSearch(
        provider="scripted",
        retrieved_at="2026-08-18T12:00:00+00:00",
        request_params={"search": "scripted"},
        total_count=len(sources),
        pages_fetched=1,
        page_identifiers=("page-1",),
        rate_limit={},
        truncated=False,
        sources=tuple(sources),
    )


#: The three retrievals, one per proposed query, in proposal order.
RETRIEVALS = (
    _retrieved(S_RECENT, S_META, S_UNCERTAIN),
    _retrieved(S_FOUNDATIONAL, S_RECENT),  # one-source overlap with query 1
    _retrieved(S_EXCLUDED),
)

QUERIES_REPLY = json.dumps(
    {
        "queries": [
            {"family": "recent", "text": "in-context learning"},
            {"family": "foundational", "text": "meta-learning"},
            {
                "family": "limitations_open_problems",
                "text": "in-context learning limitations",
            },
        ]
    }
)


def _screening_reply(**decisions: tuple[str, str]) -> str:
    """decisions maps a name to (source_id, decision)."""
    return json.dumps(
        {
            "decisions": [
                {
                    "source_id": source_id,
                    "decision": decision,
                    "reason": f"screened as {decision}",
                }
                for source_id, decision in decisions.values()
            ]
        }
    )


SCREENING_REPLY = _screening_reply(
    a=(S_RECENT.id, "relevant"),
    b=(S_META.id, "relevant"),
    c=(S_UNCERTAIN.id, "uncertain"),
    d=(S_FOUNDATIONAL.id, "relevant"),
    e=(S_EXCLUDED.id, "excluded"),
)

_EMPTY_EXTRACTION_FIELDS: dict[str, object] = {
    "datasets": [],
    "metrics": [],
    "evaluation_protocols": [],
    "baselines": [],
    "reported_results": [],
    "limitations": [],
    "future_work": [],
    "open_problems": [],
}

EXTRACT_RECENT_REPLY = json.dumps(
    {
        "source_id": S_RECENT.id,
        "support_location": "abstract",
        "sufficient_support": True,
        "insufficiency_reason": "",
        "methods": ["prompt adaptation"],
        "datasets": [
            {
                "name": "GLUE",
                "task": "language understanding",
                "version": "",
                "split": "",
                "subset": "",
                "preprocessing": "",
                "size": "",
                "availability": "unreported",
                "url": "",
                "license": "",
            }
        ],
        "metrics": ["accuracy"],
        "evaluation_protocols": ["evaluation over 3 seeds"],
        "baselines": [],
        "reported_results": ["reaches 88.5 accuracy on GLUE"],
        "limitations": [
            {
                "text": "degrades under distribution shift",
                "kind": "generalization",
            }
        ],
        "future_work": [],
        "open_problems": ["robustness under distribution shift"],
    }
)

EXTRACT_FOUNDATIONAL_REPLY = json.dumps(
    {
        "source_id": S_FOUNDATIONAL.id,
        "support_location": "abstract",
        "sufficient_support": True,
        "insufficiency_reason": "",
        "methods": ["episodic meta-learning"],
        **_EMPTY_EXTRACTION_FIELDS,
        "metrics": ["accuracy"],
        "evaluation_protocols": ["held-out task evaluation"],
    }
)

FIELD_MAP_REPLY = json.dumps(
    {
        "themes": [
            {
                "name": "Prompt adaptation",
                "summary": "Recent prompt adaptation reaching 88.5 accuracy.",
                "era": "recent",
                "source_ids": [S_RECENT.id],
            },
            {
                "name": "Meta-learning foundations",
                "summary": "Episodic training for rapid adaptation.",
                "era": "foundational",
                "source_ids": [S_FOUNDATIONAL.id],
            },
        ],
        "approaches": [
            {
                "name": "Gradient-free adaptation",
                "summary": "Adaptation without weight updates.",
                "source_ids": [S_RECENT.id],
            }
        ],
        "evaluation_practices": [
            {
                "name": "Held-out accuracy",
                "summary": "Accuracy on held-out tasks.",
                "source_ids": [S_RECENT.id, S_FOUNDATIONAL.id],
            }
        ],
        "relationships": [
            {
                "kind": "builds_on",
                "from_theme": "Prompt adaptation",
                "to_theme": "Meta-learning foundations",
                "note": "adaptation reuses episodic ideas",
            }
        ],
    }
)

INVENTORY_REPLY = json.dumps(
    {
        "problems": [
            {
                "statement": (
                    "robustness of in-context learning under distribution "
                    "shift is unresolved"
                ),
                "kind": "open_problem",
                "grounding": "the recent paper reports degradation under shift",
                "supporting_source_ids": [S_RECENT.id],
                "conflicting_source_ids": [],
            },
            {
                "statement": (
                    "whether episodic training is required for adaptation "
                    "is contested"
                ),
                "kind": "conflicting_findings",
                "grounding": (
                    "the foundational paper trains episodically; the recent "
                    "paper adapts prompts without it"
                ),
                "supporting_source_ids": [S_FOUNDATIONAL.id],
                "conflicting_source_ids": [S_RECENT.id],
            },
        ]
    }
)

HAPPY_REPLIES = (
    QUERIES_REPLY,
    SCREENING_REPLY,
    EXTRACT_RECENT_REPLY,
    EXTRACT_FOUNDATIONAL_REPLY,
    FIELD_MAP_REPLY,
    INVENTORY_REPLY,
)


def _mapper(
    tmp_path: Path,
    replies: tuple[ScriptedReply | str, ...],
    *,
    literature_outcomes: tuple[RetrievedSearch, ...] = RETRIEVALS,
) -> tuple[FieldMapper, FakeModelProvider, UsageLedger, MappingStore,
           ScriptedLiteratureProvider]:
    provider = FakeModelProvider(replies)
    ledger = UsageLedger()
    literature_provider = ScriptedLiteratureProvider(literature_outcomes)
    corpus = LiteratureCorpus(
        LiteratureStore(tmp_path / "literature"), literature_provider
    )
    store = MappingStore(tmp_path / "mapping")
    mapper = FieldMapper(
        provider=provider,
        model="fake-model",
        ledger=ledger,
        corpus=corpus,
        store=store,
        max_output_tokens=4096,
    )
    return mapper, provider, ledger, store, literature_provider


def test_the_full_run_produces_grounded_durable_records(
    tmp_path: Path,
) -> None:
    mapper, provider, ledger, _store, _ = _mapper(tmp_path, HAPPY_REPLIES)
    result = mapper.run(BRIEF)

    # Retrieval went through Task 5A with trusted dates and full provenance.
    assert [e.family.value for e in result.query_executions] == [
        "recent",
        "foundational",
        "limitations_open_problems",
    ]
    assert result.query_executions[0].from_date == "2026-01-01"
    assert result.query_executions[1].to_date == "2025-12-31"
    assert result.query_executions[1].new_unique == 1  # overlap counted
    assert all(e.search_record_id for e in result.query_executions)

    # Every screening verdict is preserved, exclusions included.
    by_source = {s.source_id: s.decision for s in result.screenings}
    assert by_source[S_EXCLUDED.id] is ScreeningDecision.EXCLUDED
    assert by_source[S_UNCERTAIN.id] is ScreeningDecision.UNCERTAIN
    assert len(result.screenings) == 5

    # The metadata-only source never reached the model: deterministic
    # insufficiency, provenance None, title-level support.
    meta = next(
        e for e in result.extractions if e.source_id == S_META.id
    )
    assert meta.sufficient_support is False
    assert meta.provenance is None
    assert meta.support_location is SupportLocation.TITLE
    assert meta.era is SourceEra.RECENT

    # The era split is trusted code's, from the brief's window.
    assert result.field_map.recent_source_ids == (S_RECENT.id,)
    assert result.field_map.foundational_source_ids == (S_FOUNDATIONAL.id,)
    assert result.field_map.themes[0].era.value == "recent"

    # Conflicting reports stay attached to the problem they contradict.
    contested = result.inventory.problems[1]
    assert contested.conflicting_source_ids == (S_RECENT.id,)

    # Coverage is bounded and honest.
    coverage = result.run_record.coverage
    assert coverage.queries_executed == 3
    assert coverage.total_retrieved == 6
    assert coverage.unique_sources == 5
    assert coverage.screened == 5
    assert coverage.relevant == 3
    assert coverage.excluded == 1
    assert coverage.uncertain == 1
    assert coverage.abstract_level == 4
    assert coverage.metadata_level == 1
    assert coverage.extraction_eligible == 3
    assert coverage.extracted == 3
    assert coverage.insufficient_support == 1
    assert coverage.saturation == 0.0  # the last query was all new

    # Exact usage accounting: the ledger and the run record agree, and
    # every call is counted exactly once.
    drained = ledger.drain()
    assert drained.calls == 6
    assert result.run_record.model_calls == 6
    assert drained.input_tokens == result.run_record.input_tokens
    assert drained.output_tokens == result.run_record.output_tokens
    assert len(provider.calls) == 6

    # Everything reloads from disk with recomputed identities.
    fresh = MappingStore(tmp_path / "mapping")
    assert fresh.get_run(result.run_record.id) == result.run_record
    assert fresh.get_field_map(result.field_map.id) == result.field_map
    assert fresh.get_inventory(result.inventory.id) == result.inventory
    assert len(fresh.screenings()) == 5
    assert len(fresh.extractions()) == 3
    assert fresh.rejected() == ()


def test_cached_replay_reruns_without_literature_network(
    tmp_path: Path,
) -> None:
    first, _, _, _, _ = _mapper(tmp_path, HAPPY_REPLIES)
    first.run(BRIEF)

    # Same literature store, a provider with nothing scripted: every
    # query must replay from the Task 5A cache.
    replay_literature = ScriptedLiteratureProvider((), name="scripted")
    corpus = LiteratureCorpus(
        LiteratureStore(tmp_path / "literature"), replay_literature
    )
    mapper = FieldMapper(
        provider=FakeModelProvider(HAPPY_REPLIES),
        model="fake-model",
        ledger=UsageLedger(),
        corpus=corpus,
        store=MappingStore(tmp_path / "mapping"),
    )
    result = mapper.run(BRIEF)
    assert replay_literature.queries == ()  # zero retrieval calls
    assert all(e.from_cache for e in result.query_executions)


def test_a_schema_violation_earns_one_bounded_correction(
    tmp_path: Path,
) -> None:
    replies = ("this is not json at all", *HAPPY_REPLIES)
    mapper, provider, ledger, store, _ = _mapper(tmp_path, replies)
    result = mapper.run(BRIEF)

    assert result.run_record.model_calls == 7
    (rejected,) = store.rejected()
    assert rejected["stage"] == "queries"
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    assert reasons[0]["rule"] == "invalid_structured_output"
    # The failed call carried no accounting; only successes reached the
    # ledger's totals — and each exactly once.
    assert ledger.drain().calls == 6
    assert len(provider.calls) == 7


def test_a_gate_rejection_earns_one_bounded_correction(
    tmp_path: Path,
) -> None:
    incomplete = _screening_reply(
        a=(S_RECENT.id, "relevant"),
        b=(S_META.id, "relevant"),
        c=(S_UNCERTAIN.id, "uncertain"),
        d=(S_FOUNDATIONAL.id, "relevant"),
        # S_EXCLUDED is missing: the gate demands a verdict for every
        # source in the batch.
    )
    replies = (
        QUERIES_REPLY,
        incomplete,
        SCREENING_REPLY,
        EXTRACT_RECENT_REPLY,
        EXTRACT_FOUNDATIONAL_REPLY,
        FIELD_MAP_REPLY,
        INVENTORY_REPLY,
    )
    mapper, _, ledger, store, _ = _mapper(tmp_path, replies)
    result = mapper.run(BRIEF)

    assert result.run_record.model_calls == 7
    (rejected,) = store.rejected()
    assert rejected["stage"] == "screening"
    reasons = rejected["reasons"]
    assert isinstance(reasons, list)
    assert {r["rule"] for r in reasons} == {"missing_decision"}
    assert all(
        s.provenance.repair_count == 1 for s in result.screenings
    )
    assert ledger.drain().calls == 7  # both calls were real and billed


def test_gate_exhaustion_fails_closed_with_everything_preserved(
    tmp_path: Path,
) -> None:
    bad = json.dumps(
        {"queries": [{"family": "recent", "text": "only one family"}]}
    )
    mapper, _, ledger, store, literature = _mapper(tmp_path, (bad, bad))
    with pytest.raises(MappingRejectedError, match="missing_family"):
        mapper.run(BRIEF)

    assert len(store.rejected()) == 2  # both attempts preserved
    assert store.runs() == ()
    assert store.screenings() == ()
    assert literature.queries == ()  # no retrieval on a refused proposal
    assert ledger.drain().calls == 2


def test_a_persistent_schema_violation_reraises_the_typed_error(
    tmp_path: Path,
) -> None:
    mapper, _, _, store, _ = _mapper(
        tmp_path, ("not json", "still not json")
    )
    with pytest.raises(StructuredOutputError):
        mapper.run(BRIEF)
    assert len(store.rejected()) == 2
    assert store.runs() == ()


def test_provider_failure_accounts_once_and_accepts_nothing(
    tmp_path: Path,
) -> None:
    billed = ProviderTransportError("upstream died").with_accounting(
        CallAccounting(
            usage=ProviderUsage(
                calls=1, input_tokens=321, output_tokens=0, model="fake"
            ),
            latency_seconds=1.5,
        )
    )
    mapper, _, ledger, store, _ = _mapper(
        tmp_path, (ScriptedReply(text="", error=billed),)
    )
    with pytest.raises(ProviderTransportError):
        mapper.run(BRIEF)

    drained = ledger.drain()
    assert drained.input_tokens == 321  # the failed call's spend, once
    assert drained.calls == 1
    assert ledger.drain().calls == 0  # and only once
    assert store.runs() == ()
    assert store.screenings() == ()
    assert store.rejected() == ()  # a failure is not a rejection
    assert store.get_brief(BRIEF.id) is not None  # the intent is durable


def test_the_model_call_budget_fails_closed(tmp_path: Path) -> None:
    tight = ResearchBrief(
        topic=BRIEF.topic,
        cutoff_date=BRIEF.cutoff_date,
        recent_window_start=BRIEF.recent_window_start,
        max_queries_per_family=1,
        results_per_query=10,
        max_screened_sources=10,
        max_extracted_sources=10,
        max_model_calls=1,
    )
    mapper, _, ledger, store, _ = _mapper(tmp_path, HAPPY_REPLIES)
    with pytest.raises(MappingBudgetError, match="budget"):
        mapper.run(tight)

    # The query proposal and its retrievals are durable; screening never
    # got its call.
    assert len(store.screenings()) == 0
    assert ledger.drain().calls == 1


def test_an_all_insufficient_corpus_is_an_honest_dead_end(
    tmp_path: Path,
) -> None:
    only_meta_relevant = _screening_reply(
        a=(S_RECENT.id, "excluded"),
        b=(S_META.id, "relevant"),
        c=(S_UNCERTAIN.id, "excluded"),
        d=(S_FOUNDATIONAL.id, "excluded"),
        e=(S_EXCLUDED.id, "excluded"),
    )
    mapper, _, _, store, _ = _mapper(
        tmp_path, (QUERIES_REPLY, only_meta_relevant)
    )
    with pytest.raises(MappingContractError, match="nothing to map"):
        mapper.run(BRIEF)

    # The honest outcome is durable: the screenings and the deterministic
    # insufficiency record exist; no map, no inventory, no run record.
    assert len(store.screenings()) == 5
    (extraction,) = store.extractions()
    assert extraction.sufficient_support is False
    assert store.runs() == ()


def test_a_literature_failure_propagates_before_any_screening(
    tmp_path: Path,
) -> None:
    mapper, _, ledger, store, _ = _mapper(
        tmp_path,
        HAPPY_REPLIES,
        literature_outcomes=(RETRIEVALS[0],),  # then the script runs dry
    )
    with pytest.raises(LiteratureProviderError):
        mapper.run(BRIEF)
    assert store.screenings() == ()
    assert ledger.drain().calls == 1  # only the query proposal was billed
