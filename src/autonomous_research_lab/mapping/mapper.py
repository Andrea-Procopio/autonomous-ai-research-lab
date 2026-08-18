"""The model-backed field mapper: a bounded brief in, a source-grounded
map and problem inventory out.

A provider-neutral service, deliberately not a role: its input is a
:class:`~.brief.ResearchBrief`, not ``ResearchState``, and its output is
literature analysis, not proposals — nothing it produces can enter the
governed commit. One run performs a fixed sequence of narrow stages::

    brief -> one structured query-proposal call     (validated, bounded)
          -> trusted retrieval via the Task 5A corpus (cache-or-live)
          -> batched structured screening calls      (every verdict kept)
          -> one structured extraction call per source with abstract
             access (metadata-only sources yield deterministic
             insufficient-support records, no model call)
          -> one structured field-map call
          -> one structured problem-inventory call
          -> deterministic coverage accounting and one run record

Authority is split the same way as everywhere else in the lab. The model
proposes query text, screening verdicts, extractions, clusters, and open
problems; trusted code derives every date range, executes every search,
stamps every era and access level, checks every payload against the
deterministic gates in :mod:`.gates`, and records everything write-once —
rejections included. A payload the gate refuses earns at most one
corrective call carrying the exact rules that fired; an inconvenient but
valid analysis has no route to a second call. Every provider call —
successful or failed — reaches the ledger exactly once and spends from
the brief's model-call budget, which fails closed when exhausted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final

from ..core.ids import occurrence_id
from ..literature.corpus import LiteratureCorpus
from ..literature.retrieval import AccessLevel, LiteratureQuery, LiteratureSource
from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OutputSchema,
    StructuredOutputError,
    UsageLedger,
)
from .brief import QueryFamily, ResearchBrief, SourceEra, classify_era
from .gates import (
    MappingRejection,
    accessible_text_of,
    check_extraction,
    check_field_map,
    check_inventory,
    check_queries,
    check_screening,
)
from .records import (
    CallProvenance,
    CoverageReport,
    DatasetAvailability,
    DatasetUse,
    ExtractionRecord,
    FieldMapRecord,
    GroupEntry,
    Limitation,
    LimitationKind,
    MappingRunRecord,
    ProblemEntry,
    ProblemInventoryRecord,
    ProblemKind,
    QueryExecution,
    RelationshipKind,
    ScreeningDecision,
    ScreeningRecord,
    SupportLocation,
    ThemeEntry,
    ThemeEra,
    ThemeRelationship,
)
from .store import MappingStore

_ABSTRACT_RENDER_CHARS: Final = 1500
"""Screening shows at most this much of each abstract — enough to judge
relevance, bounded so a batch prompt stays finite."""


class MappingContractError(RuntimeError):
    """The run cannot proceed for a deterministic reason that is not a
    model's fault: an unusable brief, or a corpus in which no source
    yielded sufficient accessible support. Durable partial records
    remain; nothing synthesized is produced."""


class MappingRejectedError(RuntimeError):
    """A model payload failed the deterministic gate and exhausted its
    corrective-call allowance. Every attempt is preserved in the store."""


class MappingBudgetError(RuntimeError):
    """The brief's model-call budget is spent. Fails closed before the
    call that would exceed it; durable partial records remain."""


# -- output contracts ---------------------------------------------------------

_FAMILIES: Final = [family.value for family in QueryFamily]

QUERY_SCHEMA: Final = OutputSchema(
    name="mapping_queries",
    json_schema={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "family": {"type": "string", "enum": _FAMILIES},
                        "text": {
                            "type": "string",
                            "description": (
                                "Plain keyword search text; no dates — "
                                "trusted code sets each family's date "
                                "range."
                            ),
                        },
                    },
                    "required": ["family", "text"],
                },
            }
        },
        "required": ["queries"],
    },
)

SCREENING_SCHEMA: Final = OutputSchema(
    name="mapping_screening",
    json_schema={
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": ["relevant", "excluded", "uncertain"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["source_id", "decision", "reason"],
                },
            }
        },
        "required": ["decisions"],
    },
)

_DATASET_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "task": {"type": "string"},
        "version": {"type": "string"},
        "split": {"type": "string"},
        "subset": {"type": "string"},
        "preprocessing": {"type": "string"},
        "size": {"type": "string"},
        "availability": {
            "type": "string",
            "enum": ["public", "private", "synthetic", "unreported"],
        },
        "url": {"type": "string"},
        "license": {"type": "string"},
    },
    "required": [
        "name",
        "task",
        "version",
        "split",
        "subset",
        "preprocessing",
        "size",
        "availability",
        "url",
        "license",
    ],
}

EXTRACTION_SCHEMA: Final = OutputSchema(
    name="mapping_extraction",
    json_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "support_location": {
                "type": "string",
                "enum": ["title", "abstract", "full_text"],
            },
            "sufficient_support": {"type": "boolean"},
            "insufficiency_reason": {"type": "string"},
            "methods": {"type": "array", "items": {"type": "string"}},
            "datasets": {"type": "array", "items": _DATASET_SCHEMA},
            "metrics": {"type": "array", "items": {"type": "string"}},
            "evaluation_protocols": {
                "type": "array",
                "items": {"type": "string"},
            },
            "baselines": {"type": "array", "items": {"type": "string"}},
            "reported_results": {
                "type": "array",
                "items": {"type": "string"},
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "compute",
                                "data",
                                "generalization",
                                "reproducibility",
                                "other",
                            ],
                        },
                    },
                    "required": ["text", "kind"],
                },
            },
            "future_work": {"type": "array", "items": {"type": "string"}},
            "open_problems": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "source_id",
            "support_location",
            "sufficient_support",
            "insufficiency_reason",
            "methods",
            "datasets",
            "metrics",
            "evaluation_protocols",
            "baselines",
            "reported_results",
            "limitations",
            "future_work",
            "open_problems",
        ],
    },
)

_GROUP_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "summary": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "summary", "source_ids"],
}

FIELD_MAP_SCHEMA: Final = OutputSchema(
    name="mapping_field_map",
    json_schema={
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "summary": {"type": "string"},
                        "era": {
                            "type": "string",
                            "enum": ["recent", "foundational", "both"],
                        },
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "summary", "era", "source_ids"],
                },
            },
            "approaches": {"type": "array", "items": _GROUP_SCHEMA},
            "evaluation_practices": {
                "type": "array",
                "items": _GROUP_SCHEMA,
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "builds_on",
                                "contrasts_with",
                                "shares_evaluation",
                            ],
                        },
                        "from_theme": {"type": "string"},
                        "to_theme": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["kind", "from_theme", "to_theme", "note"],
                },
            },
        },
        "required": [
            "themes",
            "approaches",
            "evaluation_practices",
            "relationships",
        ],
    },
)

INVENTORY_SCHEMA: Final = OutputSchema(
    name="mapping_inventory",
    json_schema={
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "open_problem",
                                "missing_comparison",
                                "missing_ablation",
                                "conflicting_findings",
                                "reproducibility_gap",
                                "data_limitation",
                                "compute_limitation",
                                "generalization_limitation",
                            ],
                        },
                        "grounding": {"type": "string"},
                        "supporting_source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "conflicting_source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "statement",
                        "kind",
                        "grounding",
                        "supporting_source_ids",
                        "conflicting_source_ids",
                    ],
                },
            }
        },
        "required": ["problems"],
    },
)


QUERY_INSTRUCTION: Final = (
    "You design focused scholarly search queries for a bounded literature "
    "mapping run. From the research brief you are shown, propose plain "
    "keyword queries (3-300 characters, no boolean operators, no dates — "
    "trusted code sets each family's date range) across these families: "
    "recent, foundational, methods, datasets_benchmarks, "
    "metrics_evaluation, baselines, limitations_open_problems. You MUST "
    "cover at least recent, foundational, and limitations_open_problems; "
    "cover other families where they serve the topic. Respect the "
    "per-family query budget stated in the brief. Never repeat a query "
    "within a family."
)

SCREENING_INSTRUCTION: Final = (
    "You screen retrieved papers for a bounded literature mapping run. "
    "For EVERY listed source, exactly once, decide from its title and "
    "accessible abstract alone whether it is relevant to the brief's "
    "topic: 'relevant' when it plainly is, 'excluded' when it plainly is "
    "not, 'uncertain' when the accessible text cannot settle it. Give a "
    "one-sentence reason per decision. Judge only what is shown; never "
    "assume content beyond the accessible text, and never claim "
    "exhaustive knowledge of the field."
)

EXTRACTION_INSTRUCTION: Final = (
    "You extract what ONE paper's accessible material actually reports, "
    "for a bounded literature mapping run. You see only bibliographic "
    "metadata and, when retrieved, the abstract — never the full text. "
    "Record only what that accessible text supports: methods, datasets "
    "(with task/version/split/subset/preprocessing/size/availability/"
    "url/license ONLY where the text itself reports them — leave every "
    "unreported detail as an empty string and availability 'unreported'), "
    "metrics, evaluation protocols, baselines, author-reported results, "
    "author-reported limitations (typed compute/data/generalization/"
    "reproducibility/other), future work, and open problems a careful "
    "reader could defend from this text. Every number you write must "
    "appear verbatim in the accessible text; every dataset name must "
    "appear verbatim. If the accessible text supports no substantive "
    "extraction, return sufficient_support=false with a short reason and "
    "every list empty — that is a valid, honest outcome. Claims that "
    "would need methods sections, tables, or appendices are not "
    "supportable here."
)

FIELD_MAP_INSTRUCTION: Final = (
    "You synthesize a field map from the per-source extractions of a "
    "bounded literature mapping run. Cluster the listed sources into "
    "themes, approaches, and evaluation practices; every cluster cites "
    "only listed source ids and every claim rests on the listed "
    "extractions. Each theme's era must follow its cited sources' era "
    "labels: 'recent' when all are recent, 'foundational' when all are "
    "foundational, 'both' otherwise. Relationships connect declared "
    "themes only. This run saw a bounded slice of the literature: never "
    "claim exhaustive coverage, a systematic review, or novelty."
)

INVENTORY_INSTRUCTION: Final = (
    "You compile a source-grounded inventory of unresolved problems from "
    "the per-source extractions and themes of a bounded literature "
    "mapping run. Each problem states one unresolved issue, typed as "
    "open_problem, missing_comparison, missing_ablation, "
    "conflicting_findings, reproducibility_gap, data_limitation, "
    "compute_limitation, or generalization_limitation; cites the listed "
    "source ids whose reported limitations, gaps, or findings ground it; "
    "and lists conflicting sources where reports disagree — preserve "
    "disagreement, never average it away. Ground every statement in what "
    "the extractions actually report; every number must appear in the "
    "cited sources' accessible text. A bounded run cannot establish "
    "novelty or completeness: never claim either."
)


# -- the service --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappingRunResult:
    """Everything one completed run produced, in memory; the same records
    are durable in the store."""

    run_record: MappingRunRecord
    brief: ResearchBrief
    query_executions: tuple[QueryExecution, ...]
    screenings: tuple[ScreeningRecord, ...]
    extractions: tuple[ExtractionRecord, ...]
    field_map: FieldMapRecord
    inventory: ProblemInventoryRecord


class _Spend:
    """Mutable per-run call and token accounting, checked against the
    brief's budget before every provider call."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


class FieldMapper:
    """See the module docstring; construction is explicit wiring, and
    every collaborator is injected — provider, ledger, corpus, store —
    so tests and live runs differ only in what is plugged in."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        ledger: UsageLedger,
        corpus: LiteratureCorpus,
        store: MappingStore,
        screening_batch_size: int = 10,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_corrective_calls: int = 1,
    ) -> None:
        if screening_batch_size < 1:
            raise ValueError("screening_batch_size must be positive")
        self._provider = provider
        self._model = model
        self._ledger = ledger
        self._corpus = corpus
        self._store = store
        self._screening_batch_size = screening_batch_size
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_corrective_calls = max_corrective_calls

    def run(self, brief: ResearchBrief) -> MappingRunResult:
        run_id = occurrence_id("map")
        self._store.record_brief(brief)
        spend = _Spend(brief.max_model_calls)

        queries_payload, _ = self._gated_call(
            self._request(
                QUERY_INSTRUCTION,
                render_brief(brief),
                QUERY_SCHEMA,
                run_id,
                "queries",
            ),
            gate=lambda payload: check_queries(payload, brief=brief),
            stage="queries",
            run_id=run_id,
            spend=spend,
        )

        executions, ordered_sources = self._retrieve(
            run_id, brief, queries_payload
        )
        to_screen = ordered_sources[: brief.max_screened_sources]
        screenings = self._screen(run_id, brief, to_screen, spend)
        decisions = {record.source_id: record for record in screenings}
        relevant = [
            source
            for source in to_screen
            if decisions[source.id].decision is ScreeningDecision.RELEVANT
        ]
        extractions = self._extract(
            run_id, brief, relevant[: brief.max_extracted_sources], spend
        )

        sufficient = tuple(r for r in extractions if r.sufficient_support)
        if not sufficient:
            raise MappingContractError(
                "no screened-relevant source yielded sufficient accessible "
                "support; there is nothing to map. The screening and "
                "extraction records are durable."
            )
        by_id = {source.id: source for source in ordered_sources}
        eras = {record.source_id: record.era for record in sufficient}
        accessible = {
            record.source_id: accessible_text_of(by_id[record.source_id])
            for record in sufficient
        }

        map_payload, map_provenance = self._gated_call(
            self._request(
                FIELD_MAP_INSTRUCTION,
                render_for_field_map(brief, sufficient, by_id),
                FIELD_MAP_SCHEMA,
                run_id,
                "field_map",
            ),
            gate=lambda payload: check_field_map(
                payload, eras=eras, accessible=accessible
            ),
            stage="field_map",
            run_id=run_id,
            spend=spend,
        )
        field_map = self._store.record_field_map(
            _field_map_record(
                map_payload, run_id, brief, eras, map_provenance
            )
        )

        inventory_payload, inventory_provenance = self._gated_call(
            self._request(
                INVENTORY_INSTRUCTION,
                render_for_inventory(brief, sufficient, field_map, by_id),
                INVENTORY_SCHEMA,
                run_id,
                "inventory",
            ),
            gate=lambda payload: check_inventory(
                payload, eras=eras, accessible=accessible
            ),
            stage="inventory",
            run_id=run_id,
            spend=spend,
        )
        inventory = self._store.record_inventory(
            _inventory_record(
                inventory_payload, run_id, brief, inventory_provenance
            )
        )

        coverage = _coverage(
            executions, ordered_sources, to_screen, screenings, relevant,
            extractions,
        )
        run_record = self._store.record_run(
            MappingRunRecord(
                run_id=run_id,
                brief_id=brief.id,
                query_execution_ids=tuple(e.id for e in executions),
                screening_ids=tuple(s.id for s in screenings),
                extraction_ids=tuple(e.id for e in extractions),
                field_map_id=field_map.id,
                inventory_id=inventory.id,
                model_calls=spend.calls,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
                coverage=coverage,
            )
        )
        return MappingRunResult(
            run_record=run_record,
            brief=brief,
            query_executions=tuple(executions),
            screenings=tuple(screenings),
            extractions=tuple(extractions),
            field_map=field_map,
            inventory=inventory,
        )

    # -- stages ----------------------------------------------------------------

    def _retrieve(
        self,
        run_id: str,
        brief: ResearchBrief,
        payload: Mapping[str, object],
    ) -> tuple[list[QueryExecution], list[LiteratureSource]]:
        """Trusted execution of the accepted queries, in payload order:
        dates from the brief, retrieval through the Task 5A corpus
        (cache-or-live), one durable execution record each."""
        executions: list[QueryExecution] = []
        ordered: list[LiteratureSource] = []
        seen: set[str] = set()
        entries = payload["queries"]
        assert isinstance(entries, Sequence)
        for entry in entries:
            assert isinstance(entry, Mapping)
            family = QueryFamily(str(entry["family"]))
            text = str(entry["text"]).strip()
            from_date, to_date = brief.date_range(family)
            query = LiteratureQuery(
                text=text,
                from_date=from_date,
                to_date=to_date,
                per_page=min(brief.results_per_query, 25),
                max_results=brief.results_per_query,
            )
            result = self._corpus.search(query)
            fresh = [s for s in result.sources if s.id not in seen]
            execution = self._store.record_query_execution(
                QueryExecution(
                    run_id=run_id,
                    family=family,
                    text=text,
                    from_date=from_date,
                    to_date=to_date,
                    query_fingerprint=query.fingerprint,
                    search_record_id=result.record.id,
                    retrieved=len(result.sources),
                    new_unique=len(fresh),
                    from_cache=result.from_cache,
                )
            )
            executions.append(execution)
            for source in fresh:
                seen.add(source.id)
                ordered.append(source)
        return executions, ordered

    def _screen(
        self,
        run_id: str,
        brief: ResearchBrief,
        sources: Sequence[LiteratureSource],
        spend: _Spend,
    ) -> list[ScreeningRecord]:
        records: list[ScreeningRecord] = []
        for start in range(0, len(sources), self._screening_batch_size):
            batch = sources[start : start + self._screening_batch_size]
            expected = [source.id for source in batch]
            payload, provenance = self._gated_call(
                self._request(
                    SCREENING_INSTRUCTION,
                    render_screening_batch(brief, batch),
                    SCREENING_SCHEMA,
                    run_id,
                    "screening",
                ),
                gate=partial(
                    check_screening, expected_source_ids=tuple(expected)
                ),
                stage="screening",
                run_id=run_id,
                spend=spend,
            )
            entries = payload["decisions"]
            assert isinstance(entries, Sequence)
            by_id = {}
            for entry in entries:
                assert isinstance(entry, Mapping)
                by_id[str(entry["source_id"])] = entry
            for source in batch:
                entry = by_id[source.id]
                records.append(
                    self._store.record_screening(
                        ScreeningRecord(
                            run_id=run_id,
                            source_id=source.id,
                            decision=ScreeningDecision(
                                str(entry["decision"])
                            ),
                            reason=str(entry["reason"]),
                            provenance=provenance,
                        )
                    )
                )
        return records

    def _extract(
        self,
        run_id: str,
        brief: ResearchBrief,
        sources: Sequence[LiteratureSource],
        spend: _Spend,
    ) -> list[ExtractionRecord]:
        records: list[ExtractionRecord] = []
        for source in sources:
            era = classify_era(source.publication_date, brief)
            if source.access_level is not AccessLevel.ABSTRACT:
                # Deterministic honesty, no model call: a title alone
                # supports no substantive extraction.
                records.append(
                    self._store.record_extraction(
                        ExtractionRecord(
                            run_id=run_id,
                            source_id=source.id,
                            era=era,
                            access_level=source.access_level.value,
                            support_location=SupportLocation.TITLE,
                            sufficient_support=False,
                            insufficiency_reason=(
                                "metadata-only access: no abstract was "
                                "retrieved, and a title alone cannot "
                                "support extraction"
                            ),
                            methods=(),
                            datasets=(),
                            metrics=(),
                            evaluation_protocols=(),
                            baselines=(),
                            reported_results=(),
                            limitations=(),
                            future_work=(),
                            open_problems=(),
                            provenance=None,
                        )
                    )
                )
                continue
            payload, provenance = self._gated_call(
                self._request(
                    EXTRACTION_INSTRUCTION,
                    render_source_for_extraction(brief, source),
                    EXTRACTION_SCHEMA,
                    run_id,
                    "extraction",
                ),
                gate=partial(check_extraction, source=source),
                stage="extraction",
                run_id=run_id,
                spend=spend,
            )
            records.append(
                self._store.record_extraction(
                    _extraction_record(payload, run_id, source, era, provenance)
                )
            )
        return records

    # -- the model call ------------------------------------------------------

    def _request(
        self,
        instruction: str,
        content: str,
        schema: OutputSchema,
        run_id: str,
        stage: str,
    ) -> ModelRequest:
        return ModelRequest(
            model=self._model,
            instruction=instruction,
            messages=(Message(role=MessageRole.USER, content=content),),
            schema=schema,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={"mapping_run": run_id, "stage": stage},
        )

    def _invoke(self, request: ModelRequest, spend: _Spend) -> ModelResponse:
        """One provider call: budget checked before, accounting reaching
        the ledger exactly once — the response on success, the attached
        cost on failure — before any error propagates."""
        if spend.calls >= spend.limit:
            raise MappingBudgetError(
                f"the brief's model-call budget ({spend.limit}) is spent; "
                f"refusing the call that would exceed it"
            )
        spend.calls += 1
        try:
            response = self._provider.invoke(request)
        except ModelProviderError as error:
            self._ledger.record_failure(error)
            raise
        self._ledger.record(response)
        spend.input_tokens += response.usage.input_tokens
        spend.output_tokens += response.usage.output_tokens
        return response

    def _attempt(
        self, request: ModelRequest, spend: _Spend
    ) -> tuple[ModelResponse | None, StructuredOutputError | None]:
        """One call whose schema violation is a correctable outcome, not
        an abort: exactly :class:`StructuredOutputError` is caught (its
        accounting already reached the ledger in ``_invoke``); every
        other provider failure propagates."""
        try:
            return self._invoke(request, spend), None
        except StructuredOutputError as error:
            return None, error

    def _gated_call(
        self,
        request: ModelRequest,
        *,
        gate: Callable[[Mapping[str, object]], tuple[MappingRejection, ...]],
        stage: str,
        run_id: str,
        spend: _Spend,
    ) -> tuple[Mapping[str, object], CallProvenance]:
        """One structured call under one deterministic gate, with the
        bounded corrective-call discipline: a schema violation and a gate
        rejection earn the same treatment — the payload is preserved, at
        most ``max_corrective_calls`` retries carry the exact rules that
        fired, and only mechanical rules — never taste — trigger the
        retry."""
        response, schema_error = self._attempt(request, spend)
        repairs = 0
        while True:
            if schema_error is not None:
                payload: Mapping[str, object] | None = None
                rejections: tuple[MappingRejection, ...] = (
                    MappingRejection(
                        "invalid_structured_output",
                        f"the reply violated the output schema: "
                        f"{schema_error}",
                    ),
                )
                fingerprint, response_id = request.fingerprint, ""
                raw: object = str(schema_error)
            else:
                assert response is not None
                payload = response.structured
                if payload is None:
                    rejections = (
                        MappingRejection(
                            "no_structured_payload",
                            "the reply carried no structured payload",
                        ),
                    )
                else:
                    rejections = gate(payload)
                fingerprint = response.request_fingerprint
                response_id = response.id
                raw = response.text
            if not rejections:
                break
            self._store.preserve_rejected(
                run_id=run_id,
                stage=stage,
                reasons=tuple((r.rule, r.detail) for r in rejections),
                request_fingerprint=fingerprint,
                response_id=response_id,
                payload=payload if payload is not None else raw,
                repair=repairs,
            )
            if repairs >= self._max_corrective_calls:
                if schema_error is not None:
                    raise schema_error
                raise MappingRejectedError(
                    f"{stage} payload rejected by the deterministic gate: "
                    + "; ".join(
                        f"{r.rule}: {r.detail}" for r in rejections
                    )
                )
            repairs += 1
            request = _repair_request(request, response, rejections, repairs)
            response, schema_error = self._attempt(request, spend)
        assert response is not None
        assert payload is not None  # an absent payload never passes the gate
        return payload, CallProvenance(
            request_fingerprint=response.request_fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=self._model,
            served_model=response.model,
            provider_request_id=response.request_id,
            latency_seconds=response.latency_seconds,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            repair_count=repairs,
        )


def _repair_request(
    base: ModelRequest,
    failed: ModelResponse | None,
    rejections: tuple[MappingRejection, ...],
    attempt: int,
) -> ModelRequest:
    """One corrective request: the failed reply (when one was parseable at
    all) plus every deterministic rule that fired. Mechanical rules only —
    never analytic taste."""
    rules = "\n".join(f"- {r.rule}: {r.detail}" for r in rejections)
    feedback = (
        f"Your output was rejected by the deterministic mapping gate. "
        f"Nothing was recorded. The rules that fired:\n{rules}\n"
        f"Return one corrected output now, satisfying every original "
        f"constraint. Ground every claim in the accessible text you were "
        f"shown; drop anything you cannot ground."
    )
    previous = (
        failed.text
        if failed is not None and failed.text
        else "(the previous reply did not satisfy the output schema)"
    )
    return ModelRequest(
        model=base.model,
        instruction=base.instruction,
        messages=(
            *base.messages,
            Message(role=MessageRole.ASSISTANT, content=previous),
            Message(role=MessageRole.USER, content=feedback),
        ),
        schema=base.schema,
        max_output_tokens=base.max_output_tokens,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        metadata={**base.metadata, "mapping_repair": str(attempt)},
    )


# -- record construction from gate-accepted payloads ---------------------------


def _extraction_record(
    payload: Mapping[str, object],
    run_id: str,
    source: LiteratureSource,
    era: SourceEra,
    provenance: CallProvenance,
) -> ExtractionRecord:
    datasets = []
    for entry in _sequence(payload["datasets"]):
        assert isinstance(entry, Mapping)
        datasets.append(
            DatasetUse(
                name=str(entry["name"]),
                task=str(entry["task"]),
                version=str(entry["version"]),
                split=str(entry["split"]),
                subset=str(entry["subset"]),
                preprocessing=str(entry["preprocessing"]),
                size=str(entry["size"]),
                availability=DatasetAvailability(str(entry["availability"])),
                url=str(entry["url"]),
                license=str(entry["license"]),
            )
        )
    limitations = []
    for entry in _sequence(payload["limitations"]):
        assert isinstance(entry, Mapping)
        limitations.append(
            Limitation(
                text=str(entry["text"]),
                kind=LimitationKind(str(entry["kind"])),
            )
        )
    return ExtractionRecord(
        run_id=run_id,
        source_id=source.id,
        era=era,
        access_level=source.access_level.value,
        support_location=SupportLocation(str(payload["support_location"])),
        sufficient_support=bool(payload["sufficient_support"]),
        insufficiency_reason=str(payload["insufficiency_reason"]),
        methods=_strings(payload["methods"]),
        datasets=tuple(datasets),
        metrics=_strings(payload["metrics"]),
        evaluation_protocols=_strings(payload["evaluation_protocols"]),
        baselines=_strings(payload["baselines"]),
        reported_results=_strings(payload["reported_results"]),
        limitations=tuple(limitations),
        future_work=_strings(payload["future_work"]),
        open_problems=_strings(payload["open_problems"]),
        provenance=provenance,
    )


def _field_map_record(
    payload: Mapping[str, object],
    run_id: str,
    brief: ResearchBrief,
    eras: Mapping[str, SourceEra],
    provenance: CallProvenance,
) -> FieldMapRecord:
    themes = []
    for entry in _sequence(payload["themes"]):
        assert isinstance(entry, Mapping)
        themes.append(
            ThemeEntry(
                name=str(entry["name"]),
                summary=str(entry["summary"]),
                era=ThemeEra(str(entry["era"])),
                source_ids=_strings(entry["source_ids"]),
            )
        )
    groups: dict[str, list[GroupEntry]] = {
        "approaches": [],
        "evaluation_practices": [],
    }
    for key, collected in groups.items():
        for entry in _sequence(payload[key]):
            assert isinstance(entry, Mapping)
            collected.append(
                GroupEntry(
                    name=str(entry["name"]),
                    summary=str(entry["summary"]),
                    source_ids=_strings(entry["source_ids"]),
                )
            )
    relationships = []
    for entry in _sequence(payload["relationships"]):
        assert isinstance(entry, Mapping)
        relationships.append(
            ThemeRelationship(
                kind=RelationshipKind(str(entry["kind"])),
                from_theme=str(entry["from_theme"]),
                to_theme=str(entry["to_theme"]),
                note=str(entry["note"]),
            )
        )
    return FieldMapRecord(
        run_id=run_id,
        brief_id=brief.id,
        themes=tuple(themes),
        approaches=tuple(groups["approaches"]),
        evaluation_practices=tuple(groups["evaluation_practices"]),
        relationships=tuple(relationships),
        recent_source_ids=tuple(
            sorted(
                source_id
                for source_id, era in eras.items()
                if era is SourceEra.RECENT
            )
        ),
        foundational_source_ids=tuple(
            sorted(
                source_id
                for source_id, era in eras.items()
                if era is SourceEra.FOUNDATIONAL
            )
        ),
        undated_source_ids=tuple(
            sorted(
                source_id
                for source_id, era in eras.items()
                if era is SourceEra.UNDATED
            )
        ),
        provenance=provenance,
    )


def _inventory_record(
    payload: Mapping[str, object],
    run_id: str,
    brief: ResearchBrief,
    provenance: CallProvenance,
) -> ProblemInventoryRecord:
    problems = []
    for entry in _sequence(payload["problems"]):
        assert isinstance(entry, Mapping)
        problems.append(
            ProblemEntry(
                statement=str(entry["statement"]),
                kind=ProblemKind(str(entry["kind"])),
                grounding=str(entry["grounding"]),
                supporting_source_ids=_strings(
                    entry["supporting_source_ids"]
                ),
                conflicting_source_ids=_strings(
                    entry["conflicting_source_ids"]
                ),
            )
        )
    return ProblemInventoryRecord(
        run_id=run_id,
        brief_id=brief.id,
        problems=tuple(problems),
        provenance=provenance,
    )


def _coverage(
    executions: Sequence[QueryExecution],
    ordered_sources: Sequence[LiteratureSource],
    screened: Sequence[LiteratureSource],
    screenings: Sequence[ScreeningRecord],
    relevant: Sequence[LiteratureSource],
    extractions: Sequence[ExtractionRecord],
) -> CoverageReport:
    by_decision = {
        decision: sum(
            1 for record in screenings if record.decision is decision
        )
        for decision in ScreeningDecision
    }
    last = executions[-1] if executions else None
    saturation = (
        1.0 - (last.new_unique / last.retrieved)
        if last is not None and last.retrieved
        else 1.0
    )
    return CoverageReport(
        queries_executed=len(executions),
        total_retrieved=sum(e.retrieved for e in executions),
        unique_sources=len(ordered_sources),
        screened=len(screened),
        screening_truncated=len(ordered_sources) - len(screened),
        relevant=by_decision[ScreeningDecision.RELEVANT],
        excluded=by_decision[ScreeningDecision.EXCLUDED],
        uncertain=by_decision[ScreeningDecision.UNCERTAIN],
        abstract_level=sum(
            1
            for source in screened
            if source.access_level is AccessLevel.ABSTRACT
        ),
        metadata_level=sum(
            1
            for source in screened
            if source.access_level is not AccessLevel.ABSTRACT
        ),
        extraction_eligible=len(relevant),
        extracted=len(extractions),
        extraction_truncated=max(0, len(relevant) - len(extractions)),
        insufficient_support=sum(
            1 for record in extractions if not record.sufficient_support
        ),
        saturation=round(saturation, 4),
    )


# -- deterministic projections ------------------------------------------------


def render_brief(brief: ResearchBrief) -> str:
    lines = [
        "## Research brief",
        f"- topic: {brief.topic}",
        f"- cutoff date: {brief.cutoff_date} (nothing after this counts)",
        f"- recent window: {brief.recent_window_start} .. "
        f"{brief.cutoff_date}",
        f"- foundational: published before {brief.recent_window_start}",
        f"- queries allowed per family: {brief.max_queries_per_family}",
    ]
    if brief.workshop_hints:
        lines.append("\n## Workshop / CFP hints (context only)")
        lines.extend(f"- {hint}" for hint in brief.workshop_hints)
    lines.append("\nPropose the focused queries now as schema JSON.")
    return "\n".join(lines)


def render_screening_batch(
    brief: ResearchBrief, sources: Sequence[LiteratureSource]
) -> str:
    lines = [
        f"## Topic\n{brief.topic}",
        "\n## Sources to screen (title + accessible abstract only)",
    ]
    for source in sources:
        lines.append(f"\n### {source.id}")
        lines.append(f"title: {source.title or '(no title reported)'}")
        lines.append(
            f"year: {source.publication_year!r}; venue: "
            f"{source.venue or '(unreported)'}; type: "
            f"{source.work_type or '(unreported)'}"
        )
        if source.abstract is None:
            lines.append("abstract: (not retrieved — metadata-only access)")
        else:
            lines.append(f"abstract: {_clip(source.abstract)}")
    lines.append(
        "\nScreen every source above, exactly once each, as schema JSON."
    )
    return "\n".join(lines)


def render_source_for_extraction(
    brief: ResearchBrief, source: LiteratureSource
) -> str:
    lines = [
        f"## Topic\n{brief.topic}",
        f"\n## Source {source.id} (access level: "
        f"{source.access_level.value})",
        f"title: {source.title or '(no title reported)'}",
        f"authors: {', '.join(source.authors) or '(unreported)'}",
        f"publication date: {source.publication_date or '(unreported)'}",
        f"venue: {source.venue or '(unreported)'}; type: "
        f"{source.work_type or '(unreported)'}",
        f"doi: {source.doi or '(none)'}; arxiv: "
        f"{source.arxiv_id or '(none)'}",
        f"\nabstract:\n{source.abstract or '(not retrieved)'}",
        "\nExtract what this accessible text supports, as schema JSON.",
    ]
    return "\n".join(lines)


def _render_extractions(
    brief: ResearchBrief,
    extractions: Sequence[ExtractionRecord],
    sources: Mapping[str, LiteratureSource],
) -> list[str]:
    lines = [
        f"## Topic\n{brief.topic}",
        "\n## Extracted sources (cite these ids and no others)",
    ]
    for record in extractions:
        source = sources[record.source_id]
        lines.append(f"\n### {record.source_id} [era: {record.era.value}]")
        lines.append(f"title: {source.title or '(no title)'}")
        lines.append(f"year: {source.publication_year!r}")
        for label, items in (
            ("methods", record.methods),
            ("datasets", tuple(d.name for d in record.datasets)),
            ("metrics", record.metrics),
            ("evaluation", record.evaluation_protocols),
            ("baselines", record.baselines),
            ("reported results", record.reported_results),
            (
                "limitations",
                tuple(
                    f"[{entry.kind.value}] {entry.text}"
                    for entry in record.limitations
                ),
            ),
            ("future work", record.future_work),
            ("open problems", record.open_problems),
        ):
            if items:
                lines.append(f"{label}: " + " | ".join(items))
    return lines


def render_for_field_map(
    brief: ResearchBrief,
    extractions: Sequence[ExtractionRecord],
    sources: Mapping[str, LiteratureSource],
) -> str:
    lines = _render_extractions(brief, extractions, sources)
    lines.append("\nSynthesize the field map now as schema JSON.")
    return "\n".join(lines)


def render_for_inventory(
    brief: ResearchBrief,
    extractions: Sequence[ExtractionRecord],
    field_map: FieldMapRecord,
    sources: Mapping[str, LiteratureSource],
) -> str:
    lines = _render_extractions(brief, extractions, sources)
    lines.append("\n## Accepted themes")
    for theme in field_map.themes:
        lines.append(
            f"- {theme.name} [{theme.era.value}]: {theme.summary} "
            f"(sources: {', '.join(theme.source_ids)})"
        )
    lines.append("\nCompile the problem inventory now as schema JSON.")
    return "\n".join(lines)


def _clip(text: str) -> str:
    if len(text) <= _ABSTRACT_RENDER_CHARS:
        return text
    return text[:_ABSTRACT_RENDER_CHARS] + " [truncated for screening]"


def _sequence(value: object) -> Sequence[object]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return value


def _strings(value: object) -> tuple[str, ...]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return tuple(str(item) for item in value)
