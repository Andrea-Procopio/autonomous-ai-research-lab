"""The field-mapping records: everything a mapping run may durably claim.

Literature analysis is not experimentation, and these records say so by
construction. Nothing here is ``Evidence``, a ``ResearchQuestion``, a
``Hypothesis``, or any other scientific-state proposition; every record
describes what external papers *report* (grounded in Task 5A source ids
and their access levels) or what the mapper *synthesized* from those
reports — and the two are never mixed in one field.

Epistemic labeling is structural, stamped by trusted code, never chosen
by a model: each record category carries exactly one claim kind (see
:data:`CLAIM_KINDS`), so "author-reported limitation" cannot drift into
"established fact" by phrasing alone. Uncertainty and conflict are first
class: screening has an ``uncertain`` verdict, extraction has an honest
insufficient-accessible-support outcome, and the problem inventory keeps
conflicting sources attached to the problem they contradict.

Identity follows the house rules: run ids are occurrences (two runs over
one brief are two events); every stored record is content-addressed over
all of its fields, provider provenance included, so identical claims from
distinct calls are distinct records and tamper is detectable on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.ids import content_id
from ..literature.retrieval import ResultOrdering
from .brief import QueryFamily, SourceEra

#: The structural epistemic label of each model-authored record category.
#: Bibliographic facts (titles, dates, venues, identifiers) live on the
#: Task 5A source records themselves and are never re-authored here.
CLAIM_KINDS: Final = {
    "screening.reason": "mapper_synthesis",
    "extraction.methods": "author_reported_claim",
    "extraction.datasets": "author_reported_claim",
    "extraction.metrics": "author_reported_claim",
    "extraction.evaluation_protocols": "author_reported_claim",
    "extraction.baselines": "author_reported_claim",
    "extraction.reported_results": "author_reported_claim",
    "extraction.limitations": "author_reported_limitation",
    "extraction.future_work": "author_reported_claim",
    "extraction.open_problems": "inferred_open_problem",
    "field_map.themes": "mapper_synthesis",
    "field_map.approaches": "mapper_synthesis",
    "field_map.evaluation_practices": "mapper_synthesis",
    "field_map.relationships": "mapper_synthesis",
    "inventory.problems": "inferred_open_problem",
}


@dataclass(frozen=True, slots=True)
class CallProvenance:
    """The complete provider provenance of one accepted model call,
    embedded verbatim in the record it produced."""

    request_fingerprint: str
    response_id: str
    provider: str
    requested_model: str
    served_model: str
    provider_request_id: str | None
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    repair_count: int


# -- screening ----------------------------------------------------------------


class ScreeningDecision(StrEnum):
    RELEVANT = "relevant"
    EXCLUDED = "excluded"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ScreeningRecord:
    """One screening verdict for one source in one run, with the
    provenance of the batch call that produced it. Every decision is
    preserved — exclusions and uncertainty included."""

    run_id: str
    source_id: str
    decision: ScreeningDecision
    reason: str
    provenance: CallProvenance
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a screening decision requires a reason")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "scrn",
                    self.run_id,
                    self.source_id,
                    self.decision,
                    self.reason,
                    self.provenance.response_id,
                ),
            )


# -- extraction ---------------------------------------------------------------


class DatasetAvailability(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    SYNTHETIC = "synthetic"
    UNREPORTED = "unreported"


@dataclass(frozen=True, slots=True)
class DatasetUse:
    """How one paper *reports* using one dataset — a record about the
    paper's text, never about the dataset itself. Nothing here implies
    the dataset was downloaded, opened, or executed. Fields the source's
    accessible text does not report stay empty rather than plausible."""

    name: str
    task: str = ""
    version: str = ""
    split: str = ""
    subset: str = ""
    preprocessing: str = ""
    size: str = ""
    availability: DatasetAvailability = DatasetAvailability.UNREPORTED
    url: str = ""
    license: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a dataset use must name the dataset")


class LimitationKind(StrEnum):
    COMPUTE = "compute"
    DATA = "data"
    GENERALIZATION = "generalization"
    REPRODUCIBILITY = "reproducibility"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Limitation:
    text: str
    kind: LimitationKind

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a limitation must carry text")


class SupportLocation(StrEnum):
    """The accessible granularity a record's claims rest on. Task 5B
    retrieves metadata and abstracts, so ``full_text`` exists only so a
    dishonest claim of it is expressible — and rejectable."""

    TITLE = "title"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"


@dataclass(frozen=True, slots=True)
class ExtractionRecord:
    """What one source's accessible material supports, or the honest
    finding that it supports too little.

    ``sufficient_support=False`` zeroes every claim category: an
    insufficiency record asserts nothing beyond its reason. ``era`` and
    ``access_level`` are stamped by trusted code from the Task 5A source;
    ``provenance`` is ``None`` exactly when the record was produced
    deterministically (a metadata-only source never reaches a model)."""

    run_id: str
    source_id: str
    era: SourceEra
    access_level: str
    support_location: SupportLocation
    sufficient_support: bool
    insufficiency_reason: str
    methods: tuple[str, ...]
    datasets: tuple[DatasetUse, ...]
    metrics: tuple[str, ...]
    evaluation_protocols: tuple[str, ...]
    baselines: tuple[str, ...]
    reported_results: tuple[str, ...]
    limitations: tuple[Limitation, ...]
    future_work: tuple[str, ...]
    open_problems: tuple[str, ...]
    provenance: CallProvenance | None
    id: str = field(default="")

    def __post_init__(self) -> None:
        claims = (
            self.methods
            or self.datasets
            or self.metrics
            or self.evaluation_protocols
            or self.baselines
            or self.reported_results
            or self.limitations
            or self.future_work
            or self.open_problems
        )
        if self.sufficient_support and not claims:
            raise ValueError(
                "an extraction claiming sufficient support must extract "
                "something; an empty one is the insufficiency outcome"
            )
        if not self.sufficient_support:
            if claims:
                raise ValueError(
                    "an insufficient-support record must not carry claims"
                )
            if not self.insufficiency_reason.strip():
                raise ValueError(
                    "an insufficient-support record must say why"
                )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "extr",
                    self.run_id,
                    self.source_id,
                    self.era,
                    self.access_level,
                    self.support_location,
                    self.sufficient_support,
                    self.insufficiency_reason,
                    self.methods,
                    tuple(
                        (
                            d.name,
                            d.task,
                            d.version,
                            d.split,
                            d.subset,
                            d.preprocessing,
                            d.size,
                            d.availability,
                            d.url,
                            d.license,
                        )
                        for d in self.datasets
                    ),
                    self.metrics,
                    self.evaluation_protocols,
                    self.baselines,
                    self.reported_results,
                    tuple((entry.text, entry.kind) for entry in self.limitations),
                    self.future_work,
                    self.open_problems,
                    (
                        self.provenance.response_id
                        if self.provenance is not None
                        else ""
                    ),
                ),
            )


# -- the field map ------------------------------------------------------------


class ThemeEra(StrEnum):
    """A theme's position on the brief's time axis — checked against the
    trusted era classification of the sources it cites."""

    RECENT = "recent"
    FOUNDATIONAL = "foundational"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class ThemeEntry:
    name: str
    summary: str
    era: ThemeEra
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupEntry:
    """A named cluster (an approach, an evaluation practice) with the
    sources that ground it."""

    name: str
    summary: str
    source_ids: tuple[str, ...]


class RelationshipKind(StrEnum):
    BUILDS_ON = "builds_on"
    CONTRASTS_WITH = "contrasts_with"
    SHARES_EVALUATION = "shares_evaluation"


@dataclass(frozen=True, slots=True)
class ThemeRelationship:
    kind: RelationshipKind
    from_theme: str
    to_theme: str
    note: str


@dataclass(frozen=True, slots=True)
class FieldMapRecord:
    """The synthesized map of the field: themes, approaches, evaluation
    practices, and relationships — every entry grounded in extracted
    source ids — plus the deterministic era partition of the extracted
    corpus, stamped by trusted code."""

    run_id: str
    brief_id: str
    themes: tuple[ThemeEntry, ...]
    approaches: tuple[GroupEntry, ...]
    evaluation_practices: tuple[GroupEntry, ...]
    relationships: tuple[ThemeRelationship, ...]
    recent_source_ids: tuple[str, ...]
    foundational_source_ids: tuple[str, ...]
    undated_source_ids: tuple[str, ...]
    provenance: CallProvenance
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.themes:
            raise ValueError("a field map requires at least one theme")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "fmap",
                    self.run_id,
                    self.brief_id,
                    tuple(
                        (t.name, t.summary, t.era, t.source_ids)
                        for t in self.themes
                    ),
                    tuple(
                        (g.name, g.summary, g.source_ids)
                        for g in self.approaches
                    ),
                    tuple(
                        (g.name, g.summary, g.source_ids)
                        for g in self.evaluation_practices
                    ),
                    tuple(
                        (r.kind, r.from_theme, r.to_theme, r.note)
                        for r in self.relationships
                    ),
                    self.recent_source_ids,
                    self.foundational_source_ids,
                    self.undated_source_ids,
                    self.provenance.response_id,
                ),
            )


# -- the problem inventory ----------------------------------------------------


class ProblemKind(StrEnum):
    OPEN_PROBLEM = "open_problem"
    MISSING_COMPARISON = "missing_comparison"
    MISSING_ABLATION = "missing_ablation"
    CONFLICTING_FINDINGS = "conflicting_findings"
    REPRODUCIBILITY_GAP = "reproducibility_gap"
    DATA_LIMITATION = "data_limitation"
    COMPUTE_LIMITATION = "compute_limitation"
    GENERALIZATION_LIMITATION = "generalization_limitation"


@dataclass(frozen=True, slots=True)
class ProblemEntry:
    """One source-grounded unresolved problem. ``supporting_source_ids``
    are the papers whose reported limitations, gaps, or findings ground
    it; ``conflicting_source_ids`` are papers whose reports pull the
    other way — preserved, never averaged away."""

    statement: str
    kind: ProblemKind
    grounding: str
    supporting_source_ids: tuple[str, ...]
    conflicting_source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("a problem requires a statement")
        if not self.supporting_source_ids:
            raise ValueError(
                "a problem must cite at least one supporting source"
            )


@dataclass(frozen=True, slots=True)
class ProblemInventoryRecord:
    run_id: str
    brief_id: str
    problems: tuple[ProblemEntry, ...]
    provenance: CallProvenance
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.problems:
            raise ValueError(
                "an inventory requires at least one problem; a mapping run "
                "that found none reports that through its coverage record, "
                "not through an empty inventory"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "pinv",
                    self.run_id,
                    self.brief_id,
                    tuple(
                        (
                            p.statement,
                            p.kind,
                            p.grounding,
                            p.supporting_source_ids,
                            p.conflicting_source_ids,
                        )
                        for p in self.problems
                    ),
                    self.provenance.response_id,
                ),
            )


# -- retrieval and coverage ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryExecution:
    """One validated model-proposed query, executed by trusted code
    through the Task 5A corpus: family, exact text, the trusted date
    range and retrieval strategy, the literature-layer fingerprints, and
    what it returned."""

    run_id: str
    family: QueryFamily
    text: str
    from_date: str
    to_date: str
    query_fingerprint: str
    search_record_id: str
    retrieved: int
    new_unique: int
    from_cache: bool
    ordering: ResultOrdering = ResultOrdering.RECENCY
    """The retrieval strategy trusted code selected for this family —
    a discovery signal, never a quality claim. Joins the identity and
    the payload only away from the recency default, so records written
    before strategies existed still verify."""

    refinement_round: int = 0
    """0 for the initial proposal; N for the Nth bounded refinement."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            parts: tuple[object, ...] = (
                self.run_id,
                self.family,
                self.text,
                self.from_date,
                self.to_date,
                self.query_fingerprint,
                self.search_record_id,
                self.retrieved,
                self.new_unique,
                self.from_cache,
            )
            if (
                self.ordering is not ResultOrdering.RECENCY
                or self.refinement_round
            ):
                parts = (*parts, self.ordering, self.refinement_round)
            object.__setattr__(self, "id", content_id("qrun", *parts))


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Bounded-coverage bookkeeping, computed by trusted code. It exists
    to keep the map honest about what was *not* covered: truncations are
    counted, saturation is an indicator (the fraction of the final
    query's results already seen — high means queries were finding the
    same papers, low means coverage was still growing), and nothing in
    this record can claim exhaustiveness."""

    queries_executed: int
    total_retrieved: int
    unique_sources: int
    screened: int
    screening_truncated: int
    relevant: int
    excluded: int
    uncertain: int
    abstract_level: int
    metadata_level: int
    extraction_eligible: int
    extracted: int
    extraction_truncated: int
    insufficient_support: int
    saturation: float
    """New-unique fraction is (1 - saturation) for the final query; 1.0
    means the last query returned nothing unseen."""


@dataclass(frozen=True, slots=True)
class MappingRunRecord:
    """The completed run: what was asked, what was spent, and the ids of
    everything produced. Written once, after the inventory."""

    run_id: str
    brief_id: str
    query_execution_ids: tuple[str, ...]
    screening_ids: tuple[str, ...]
    extraction_ids: tuple[str, ...]
    field_map_id: str
    inventory_id: str
    model_calls: int
    input_tokens: int
    output_tokens: int
    coverage: CoverageReport
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "mrun",
                    self.run_id,
                    self.brief_id,
                    self.query_execution_ids,
                    self.screening_ids,
                    self.extraction_ids,
                    self.field_map_id,
                    self.inventory_id,
                    self.model_calls,
                    self.input_tokens,
                    self.output_tokens,
                    tuple(
                        getattr(self.coverage, name)
                        for name in (
                            "queries_executed",
                            "total_retrieved",
                            "unique_sources",
                            "screened",
                            "screening_truncated",
                            "relevant",
                            "excluded",
                            "uncertain",
                            "abstract_level",
                            "metadata_level",
                            "extraction_eligible",
                            "extracted",
                            "extraction_truncated",
                            "insufficient_support",
                            "saturation",
                        )
                    ),
                ),
            )
