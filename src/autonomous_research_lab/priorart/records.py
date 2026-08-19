"""The prior-art challenge records: everything a challenge run may
durably claim.

A prior-art comparison is an artifact-grounded judgment about *this
bounded corpus* — never a fact about the world's literature, never
scientific state, and never a novelty certificate. Nothing here is
``Evidence``, a ``Hypothesis``, or any other scientific-state
proposition; and nothing here touches the candidate records it
challenges — the portfolio stays immutable, and the verdict lives in
this package's own :class:`~.assessment.PriorArtAssessment`.

Epistemic labeling is structural, stamped by trusted code, never chosen
by a model (see :data:`CLAIM_KINDS`): a screening decision and a
dimension comparison are judgments grounded in a source's accessible
text — the gate holds every snippet to that text verbatim — while every
count, date range, ordering, access level, and the verdict itself are
computed by trusted code. Citation counts order retrieval; they are
never evidence of correctness or relevance. Absence from this corpus is
never proof of novelty.

Identity follows the house rules: run ids are occurrences, every stored
record is content-addressed over all of its fields, provider provenance
included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.ids import content_id
from ..literature.retrieval import ResultOrdering
from ..mapping.records import CallProvenance, SupportLocation

#: The structural epistemic label of each model-authored record category.
#: Everything else on these records — counts, dates, access levels,
#: orderings, the known-prior-art flag, and the verdict — is trusted-code
#: computation and carries no model authorship at all.
CLAIM_KINDS: Final = {
    "query.text": "search_proposal",
    "screen.decision": "artifact_grounded_judgment",
    "screen.reason": "artifact_grounded_judgment",
    "comparison.candidate_position": "artifact_grounded_judgment",
    "comparison.prior_work_position": "artifact_grounded_judgment",
    "comparison.overlap_features": "artifact_grounded_judgment",
    "comparison.material_differences": "artifact_grounded_judgment",
    "comparison.similarity": "artifact_grounded_judgment",
}


class PriorArtQueryFamily(StrEnum):
    """The six fixed angles every candidate is searched from. The model
    proposes each family's text; trusted code owns the date range and
    the retrieval strategy."""

    MECHANISM = "mechanism"
    PROBLEM_MECHANISM = "problem_mechanism"
    EVALUATION_SETUP = "evaluation_setup"
    SYNONYMS_LEGACY = "synonyms_legacy"
    COMPETING_APPROACHES = "competing_approaches"
    RECENT = "recent"


@dataclass(frozen=True, slots=True)
class PriorArtQueryExecution:
    """One validated model-proposed query for one candidate, executed by
    trusted code through the Task 5A corpus: family, exact text, the
    trusted date range and retrieval strategy, the literature-layer
    fingerprints, and what it returned."""

    run_id: str
    candidate_id: str
    family: PriorArtQueryFamily
    text: str
    from_date: str
    to_date: str
    query_fingerprint: str
    search_record_id: str
    retrieved: int
    new_unique: int
    from_cache: bool
    ordering: ResultOrdering
    plan_groups: tuple[tuple[str, ...], ...] = ()
    """The canonical concept groups the trusted renderer built ``text``
    from — groups conjoined, alternatives within a group as
    alternatives. Empty on pre-5D.1 records, whose ``text`` was the
    model's own string; both join the identity only when present, so
    the old records still verify."""

    renderer: str = ""
    """The rendering-scheme version that produced ``text`` from
    ``plan_groups``; empty on pre-5D.1 records."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a query execution records the executed text")
        if bool(self.plan_groups) != bool(self.renderer):
            raise ValueError(
                "a planned execution names its renderer, and a renderer "
                "implies a plan"
            )
        if not self.id:
            parts: tuple[object, ...] = (
                self.run_id,
                self.candidate_id,
                self.family,
                self.text,
                self.from_date,
                self.to_date,
                self.query_fingerprint,
                self.search_record_id,
                self.retrieved,
                self.new_unique,
                self.from_cache,
                self.ordering,
            )
            if self.plan_groups or self.renderer:
                parts = (*parts, self.plan_groups, self.renderer)
            object.__setattr__(self, "id", content_id("pqx", *parts))


class SimilarityDecision(StrEnum):
    """One source's screened relation to one candidate — similarity to
    *this* candidate, not topical relevance. ``UNDECIDABLE`` is honest:
    the accessible text does not settle the question."""

    POTENTIAL_OVERLAP = "potential_overlap"
    RELATED = "related"
    UNRELATED = "unrelated"
    UNDECIDABLE = "undecidable"


@dataclass(frozen=True, slots=True)
class PriorArtScreeningRecord:
    """One screening judgment for one source against one candidate, with
    the provenance of the batch call that produced it. Every decision is
    preserved — unrelated and undecidable included."""

    run_id: str
    candidate_id: str
    source_id: str
    known_prior_art: bool
    """Stamped by trusted code: whether this source resolves to one the
    candidate itself cited. Never model-asserted."""

    decision: SimilarityDecision
    reason: str
    provenance: CallProvenance
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a screening judgment requires a reason")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "pscr",
                    self.run_id,
                    self.candidate_id,
                    self.source_id,
                    self.known_prior_art,
                    self.decision,
                    self.reason,
                    self.provenance.response_id,
                ),
            )


class ComparisonDimension(StrEnum):
    """The five explicit dimensions every nearest-work comparison must
    cover — a comparison that skips one cannot be recorded."""

    SCIENTIFIC_QUESTION = "scientific_question"
    MECHANISM = "mechanism"
    DATA_SETTING = "data_setting"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    CLAIMED_CONTRIBUTION = "claimed_contribution"


DIMENSIONS: Final = tuple(ComparisonDimension)


class SimilarityLabel(StrEnum):
    """The model's overall reading of one compared work — an
    artifact-grounded judgment the deterministic verdict consumes but
    never trusts alone: a ``SUBSTANTIAL_MATCH`` counts only because the
    gate already forced every dimension to ground itself in the source's
    accessible text."""

    SUBSTANTIAL_MATCH = "substantial_match"
    RELATED = "related"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class DimensionComparison:
    """One dimension of one candidate-versus-prior-work comparison: what
    the candidate proposes, what the prior work's accessible text
    reports, and the verbatim snippet of that text the reading rests
    on. The snippet is the support location made falsifiable — the gate
    re-finds it in the named part of the source or rejects the
    comparison."""

    dimension: ComparisonDimension
    candidate_position: str
    prior_work_position: str
    support_location: SupportLocation
    support_snippet: str

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate_position", self.candidate_position),
            ("prior_work_position", self.prior_work_position),
            ("support_snippet", self.support_snippet),
        ):
            if not value.strip():
                raise ValueError(
                    f"a dimension comparison requires {label}"
                )


@dataclass(frozen=True, slots=True)
class WorkComparison:
    """One gated comparison of one candidate against one prior work,
    across all five dimensions. The coherence rules are structural: a
    substantial match without named overlapping features, or a
    distinction without named material differences, cannot be
    recorded."""

    run_id: str
    candidate_id: str
    source_id: str
    known_prior_art: bool
    dimensions: tuple[DimensionComparison, ...]
    overlap_features: tuple[str, ...]
    material_differences: tuple[str, ...]
    similarity: SimilarityLabel
    provenance: CallProvenance
    id: str = field(default="")

    def __post_init__(self) -> None:
        covered = tuple(entry.dimension for entry in self.dimensions)
        if sorted(covered) != sorted(DIMENSIONS):
            raise ValueError(
                "a comparison covers each of the five dimensions exactly "
                "once"
            )
        if (
            self.similarity is not SimilarityLabel.DISTINCT
            and not self.overlap_features
        ):
            raise ValueError(
                f"a {self.similarity.value} comparison names the "
                f"overlapping features it rests on"
            )
        if (
            self.similarity is not SimilarityLabel.SUBSTANTIAL_MATCH
            and not self.material_differences
        ):
            raise ValueError(
                f"a {self.similarity.value} comparison names the material "
                f"differences it rests on"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "pcmp",
                    self.run_id,
                    self.candidate_id,
                    self.source_id,
                    self.known_prior_art,
                    tuple(
                        (
                            entry.dimension,
                            entry.candidate_position,
                            entry.prior_work_position,
                            entry.support_location,
                            entry.support_snippet,
                        )
                        for entry in self.dimensions
                    ),
                    self.overlap_features,
                    self.material_differences,
                    self.similarity,
                    self.provenance.response_id,
                ),
            )


@dataclass(frozen=True, slots=True)
class PriorArtCoverage:
    """Bounded-search bookkeeping for one candidate, computed by trusted
    code alone. It exists to keep the verdict honest about what was
    *not* covered: exclusions, truncations, and undecidable screens are
    counted, saturation is an indicator (the fraction of the final
    query's results already seen), and nothing in this record can claim
    exhaustiveness."""

    families_executed: tuple[str, ...]
    queries_executed: int
    total_retrieved: int
    unique_sources: int
    """The deduplicated pool: fresh retrieval plus the candidate's own
    cited sources, injected so the challenge can compare against the
    prior art the candidate already knows."""

    overlap: int
    saturation: float
    post_cutoff_excluded: int
    undated_sources: int
    abstract_level: int
    metadata_level: int
    known_prior_art_listed: int
    known_prior_art_recovered: int
    """How many cited sources the fresh searches surfaced on their own —
    a retrieval-quality signal, not a verdict input."""

    screened: int
    potential_overlap: int
    related: int
    unrelated: int
    undecidable: int
    metadata_ambiguous: int
    """METADATA-access sources screened potentially overlapping or
    undecidable: possible direct prior art with no abstract to compare.
    Each one blocks a DISTINGUISHED verdict."""

    screening_truncated: int
    compared_works: int

    def __post_init__(self) -> None:
        if len(set(self.families_executed)) != len(self.families_executed):
            raise ValueError("families_executed lists each family once")
        if self.queries_executed < len(self.families_executed):
            raise ValueError(
                "queries_executed cannot undercount the executed families"
            )
        counts = (
            self.queries_executed,
            self.total_retrieved,
            self.unique_sources,
            self.overlap,
            self.post_cutoff_excluded,
            self.undated_sources,
            self.abstract_level,
            self.metadata_level,
            self.known_prior_art_listed,
            self.known_prior_art_recovered,
            self.screened,
            self.potential_overlap,
            self.related,
            self.unrelated,
            self.undecidable,
            self.metadata_ambiguous,
            self.screening_truncated,
            self.compared_works,
        )
        if min(counts) < 0:
            raise ValueError("coverage counts cannot be negative")
        if not 0.0 <= self.saturation <= 1.0:
            raise ValueError("saturation is a fraction in 0..1")
        if self.overlap != (
            self.total_retrieved
            + self.known_prior_art_listed
            - self.unique_sources
        ):
            raise ValueError(
                "overlap must account exactly for the appearances beyond "
                "each source's first"
            )
        screenable = self.unique_sources - self.post_cutoff_excluded
        if self.abstract_level + self.metadata_level != screenable:
            raise ValueError(
                "the access-level split must partition the in-cutoff pool"
            )
        if self.screened + self.screening_truncated != screenable:
            raise ValueError(
                "screened and truncated must partition the in-cutoff pool"
            )
        decisions = (
            self.potential_overlap
            + self.related
            + self.unrelated
            + self.undecidable
        )
        if decisions != self.screened:
            raise ValueError(
                "the screening decisions must partition the screened pool"
            )
        if self.metadata_ambiguous > self.metadata_level:
            raise ValueError(
                "metadata_ambiguous cannot exceed the metadata-level pool"
            )
        if self.metadata_ambiguous > self.potential_overlap + self.undecidable:
            raise ValueError(
                "metadata_ambiguous counts only potentially overlapping or "
                "undecidable screens"
            )
        if self.compared_works > self.screened:
            raise ValueError("only screened works can be compared")
        if self.known_prior_art_recovered > self.known_prior_art_listed:
            raise ValueError(
                "recovered cited sources cannot exceed those listed"
            )


@dataclass(frozen=True, slots=True)
class PriorArtRunRecord:
    """The completed challenge: what was asked, what was spent, the full
    lineage back to the CFP, and the ids of everything produced — one
    assessment per candidate, in candidate order. Written once, after
    the last assessment."""

    run_id: str
    directive_id: str
    ideation_run_record_id: str
    ideation_run_id: str
    assessment_id: str
    map_run_id: str
    snapshot_id: str
    candidate_ids: tuple[str, ...]
    prior_art_assessment_ids: tuple[str, ...]
    query_execution_ids: tuple[str, ...]
    screening_ids: tuple[str, ...]
    comparison_ids: tuple[str, ...]
    model_calls: int
    input_tokens: int
    output_tokens: int
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.candidate_ids:
            raise ValueError(
                "a challenge run names the candidates it assessed; a "
                "refusal portfolio never enters"
            )
        if len(self.prior_art_assessment_ids) != len(self.candidate_ids):
            raise ValueError(
                "every candidate carries exactly one assessment, in "
                "candidate order"
            )
        for label, items in (
            ("candidate_ids", self.candidate_ids),
            ("prior_art_assessment_ids", self.prior_art_assessment_ids),
            ("query_execution_ids", self.query_execution_ids),
            ("screening_ids", self.screening_ids),
            ("comparison_ids", self.comparison_ids),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"{label} lists each id once")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "prun",
                    self.run_id,
                    self.directive_id,
                    self.ideation_run_record_id,
                    self.ideation_run_id,
                    self.assessment_id,
                    self.map_run_id,
                    self.snapshot_id,
                    self.candidate_ids,
                    self.prior_art_assessment_ids,
                    self.query_execution_ids,
                    self.screening_ids,
                    self.comparison_ids,
                    self.model_calls,
                    self.input_tokens,
                    self.output_tokens,
                ),
            )
