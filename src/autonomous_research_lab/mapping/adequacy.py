"""The deterministic answer to one question: is this literature map
sufficiently grounded for bounded idea generation?

"Adequate" here means adequate *for Task 5C's bounded candidate
generation under this brief* — never exhaustive coverage, never a
systematic review, never proof of novelty. "Not found in this bounded
corpus" means exactly that. And insufficiency is a successful,
scientifically valid outcome: a run that honestly reports
``INSUFFICIENT_COVERAGE`` has done its job.

The verdict is computed by trusted code alone, from accepted durable
records — screenings, extractions, the field map, the inventory, the
coverage accounting — under explicit, configurable thresholds that are
embedded in the assessment record itself, so a reloaded verdict carries
the exact bar it was measured against. No model call participates, no
single metric (source count included) is sufficient on its own, and
every insufficiency is a typed reason a later stage can act on.

Support tiers keep single-paper reports in their place: a limitation one
paper reports stays in the inventory as exactly that — never promoted to
field-wide consensus — while a problem independently supported by
multiple distinct sources, or contradicted between sources, is labeled
as such. Citation counts appear nowhere here: they ordered retrieval,
and ordering is not evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ..core.ids import content_id
from .brief import SourceEra
from .records import (
    CoverageReport,
    ExtractionRecord,
    FieldMapRecord,
    ProblemEntry,
    ProblemInventoryRecord,
    ProblemKind,
    ScreeningDecision,
    ScreeningRecord,
)

if TYPE_CHECKING:
    from .store import MappingStore


class AdequacyStatus(StrEnum):
    ADEQUATE_FOR_IDEA_GENERATION = "adequate_for_idea_generation"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"


class AdequacyReasonCode(StrEnum):
    """The typed vocabulary of insufficiency. Every failing rule speaks
    one of these, so a consumer can act on the verdict mechanically."""

    TOO_FEW_RELEVANT = "too_few_relevant"
    TOO_FEW_GROUNDED = "too_few_grounded"
    FAMILY_COVERAGE_THIN = "family_coverage_thin"
    RECENT_COVERAGE_THIN = "recent_coverage_thin"
    FOUNDATIONAL_COVERAGE_THIN = "foundational_coverage_thin"
    EXCESSIVE_UNCERTAINTY = "excessive_uncertainty"
    THEME_SUPPORT_THIN = "theme_support_thin"
    PROBLEM_SUPPORT_THIN = "problem_support_thin"
    CROSS_PAPER_PATTERN_UNSUPPORTED = "cross_paper_pattern_unsupported"


@dataclass(frozen=True, slots=True)
class AdequacyReason:
    """One deterministic rule that found the map insufficient."""

    code: AdequacyReasonCode
    detail: str


class SupportTier(StrEnum):
    """How well one inventory problem is actually supported — computed,
    never claimed."""

    SINGLE_SOURCE_LIMITATION = "single_source_limitation"
    """A limitation one paper reports about its own work. It belongs in
    the inventory as that — never as field-wide consensus."""

    TENTATIVE = "tentative"
    """An inferred problem resting on a single source."""

    MULTI_SOURCE = "multi_source"
    """Independently supported by two or more distinct sources."""

    CONTRADICTED = "contradicted"
    """Sources disagree; the disagreement is preserved, not averaged."""


#: Problem kinds that are one paper's report about its own boundaries
#: when single-sourced (rather than an inference about the field).
_LIMITATION_KINDS: Final = frozenset(
    {
        ProblemKind.DATA_LIMITATION,
        ProblemKind.COMPUTE_LIMITATION,
        ProblemKind.GENERALIZATION_LIMITATION,
        ProblemKind.REPRODUCIBILITY_GAP,
    }
)

#: Problem kinds whose very claim is about a relationship *between*
#: papers, and therefore cannot rest on one source.
_CROSS_PAPER_KINDS: Final = frozenset({ProblemKind.CONFLICTING_FINDINGS})


def support_tier(problem: ProblemEntry) -> SupportTier:
    """The deterministic support tier of one inventory problem."""
    if problem.conflicting_source_ids:
        return SupportTier.CONTRADICTED
    if len(set(problem.supporting_source_ids)) >= 2:
        return SupportTier.MULTI_SOURCE
    if problem.kind in _LIMITATION_KINDS:
        return SupportTier.SINGLE_SOURCE_LIMITATION
    return SupportTier.TENTATIVE


@dataclass(frozen=True, slots=True)
class ProblemSupport:
    """One problem's computed support standing, kept with the assessment
    so the tier survives reload alongside the verdict."""

    statement: str
    kind: ProblemKind
    tier: SupportTier
    distinct_supporting: int
    conflicting: int


@dataclass(frozen=True, slots=True)
class AdequacyThresholds:
    """The explicit bar. Every value is deliberate, configurable where
    scientifically appropriate, validated at construction, and recorded
    verbatim inside each assessment — no constant hides in code.

    Defaults are set from the Task 5B live evidence: five grounded
    sources carried ten problems, which was too thin a base for
    candidate generation.
    """

    min_relevant_sources: int = 8
    min_grounded_sources: int = 6
    """Sources whose extraction found sufficient accessible support —
    the material the map and inventory actually cite."""

    min_families_with_relevant: int = 3
    """Distinct query families that contributed at least one relevant
    source: coverage must come from more than one retrieval angle."""

    min_recent_grounded: int = 2
    min_foundational_grounded: int = 2
    max_uncertain_fraction: float = 0.34
    """Screened sources the screener could not settle, as a fraction; a
    corpus this undecidable needs better retrieval, not more inference."""

    min_multi_source_themes: int = 1
    """At least this many field-map themes must rest on two or more
    distinct sources — otherwise the map has no cross-paper structure."""

    min_multi_source_problems: int = 1
    """At least this many inventory problems must be independently
    multi-source (or contradicted, which is also cross-paper)."""

    def __post_init__(self) -> None:
        for entry in fields(self):
            value = getattr(self, entry.name)
            if entry.name == "max_uncertain_fraction":
                if not 0.0 < value <= 1.0:
                    raise ValueError(
                        "max_uncertain_fraction must be in (0, 1]"
                    )
            elif int(value) < 1:
                raise ValueError(f"{entry.name} must be positive")


@dataclass(frozen=True, slots=True)
class AdequacyMetrics:
    """What was measured, kept apart from what was concluded. Every
    number derives from durable records; none is sufficient alone."""

    screened: int
    relevant_sources: int
    excluded_sources: int
    uncertain_sources: int
    uncertain_fraction: float
    grounded_sources: int
    insufficient_extractions: int
    metadata_only_relevant: int
    recent_grounded: int
    foundational_grounded: int
    undated_grounded: int
    families_with_relevant: tuple[str, ...]
    total_retrieved: int
    unique_sources: int
    overlap: int
    saturation: float
    screening_truncated: int
    extraction_truncated: int
    multi_source_themes: int
    single_source_themes: int
    multi_source_problems: int
    tentative_problems: int
    single_source_limitation_problems: int
    contradicted_problems: int


@dataclass(frozen=True, slots=True)
class MapAdequacyAssessment:
    """The durable verdict: status, typed reasons, the thresholds it was
    measured against, the metrics it rests on, and every problem's
    support tier. Content-addressed over all of it, so a reloaded
    assessment provably carries the same verdict and the same reasons."""

    run_id: str
    brief_id: str
    field_map_id: str
    inventory_id: str
    status: AdequacyStatus
    reasons: tuple[AdequacyReason, ...]
    thresholds: AdequacyThresholds
    metrics: AdequacyMetrics
    problem_support: tuple[ProblemSupport, ...]
    id: str = field(default="")

    def __post_init__(self) -> None:
        if (self.status is AdequacyStatus.INSUFFICIENT_COVERAGE) != bool(
            self.reasons
        ):
            raise ValueError(
                "insufficiency and reasons come together: an adequate "
                "assessment carries none, an insufficient one at least one"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "madq",
                    self.run_id,
                    self.brief_id,
                    self.field_map_id,
                    self.inventory_id,
                    self.status,
                    tuple((r.code, r.detail) for r in self.reasons),
                    tuple(
                        getattr(self.thresholds, entry.name)
                        for entry in fields(AdequacyThresholds)
                    ),
                    tuple(
                        getattr(self.metrics, entry.name)
                        for entry in fields(AdequacyMetrics)
                    ),
                    tuple(
                        (
                            p.statement,
                            p.kind,
                            p.tier,
                            p.distinct_supporting,
                            p.conflicting,
                        )
                        for p in self.problem_support
                    ),
                ),
            )


class InadequateFieldMapError(RuntimeError):
    """Raised by the Task 5C guard when the assessed map is not adequate
    for idea generation. Carries the typed reasons."""

    def __init__(
        self, message: str, *, reasons: tuple[AdequacyReason, ...] = ()
    ) -> None:
        super().__init__(message)
        self.reasons = reasons


def assess_adequacy(
    *,
    run_id: str,
    brief_id: str,
    screenings: Sequence[ScreeningRecord],
    extractions: Sequence[ExtractionRecord],
    field_map: FieldMapRecord,
    inventory: ProblemInventoryRecord,
    family_sources: Mapping[str, Sequence[str]],
    coverage: CoverageReport,
    thresholds: AdequacyThresholds,
) -> MapAdequacyAssessment:
    """The deterministic verdict, from durable records alone.

    ``family_sources`` maps each executed query family to the source ids
    its searches returned — derived from the durable query-execution and
    literature search records, so the family-coverage rule can tell
    which retrieval angles actually produced relevant material. Fails
    closed: any firing rule yields ``INSUFFICIENT_COVERAGE``.
    """
    relevant_ids = {
        record.source_id
        for record in screenings
        if record.decision is ScreeningDecision.RELEVANT
    }
    excluded = sum(
        1
        for record in screenings
        if record.decision is ScreeningDecision.EXCLUDED
    )
    uncertain = sum(
        1
        for record in screenings
        if record.decision is ScreeningDecision.UNCERTAIN
    )
    screened = len(screenings)
    uncertain_fraction = round(uncertain / screened, 4) if screened else 0.0

    grounded = [record for record in extractions if record.sufficient_support]
    insufficient = [
        record for record in extractions if not record.sufficient_support
    ]
    metadata_only = sum(
        1 for record in insufficient if record.access_level != "abstract"
    )
    eras = {
        era: sum(1 for record in grounded if record.era is era)
        for era in SourceEra
    }
    families_with_relevant = tuple(
        sorted(
            family
            for family, source_ids in family_sources.items()
            if relevant_ids & set(source_ids)
        )
    )

    theme_support = [
        len(set(theme.source_ids)) for theme in field_map.themes
    ]
    multi_source_themes = sum(1 for count in theme_support if count >= 2)
    single_source_themes = sum(1 for count in theme_support if count == 1)

    problem_support = tuple(
        ProblemSupport(
            statement=problem.statement,
            kind=problem.kind,
            tier=support_tier(problem),
            distinct_supporting=len(set(problem.supporting_source_ids)),
            conflicting=len(set(problem.conflicting_source_ids)),
        )
        for problem in inventory.problems
    )
    tiers = {
        tier: sum(1 for p in problem_support if p.tier is tier)
        for tier in SupportTier
    }

    metrics = AdequacyMetrics(
        screened=screened,
        relevant_sources=len(relevant_ids),
        excluded_sources=excluded,
        uncertain_sources=uncertain,
        uncertain_fraction=uncertain_fraction,
        grounded_sources=len(grounded),
        insufficient_extractions=len(insufficient),
        metadata_only_relevant=metadata_only,
        recent_grounded=eras[SourceEra.RECENT],
        foundational_grounded=eras[SourceEra.FOUNDATIONAL],
        undated_grounded=eras[SourceEra.UNDATED],
        families_with_relevant=families_with_relevant,
        total_retrieved=coverage.total_retrieved,
        unique_sources=coverage.unique_sources,
        overlap=coverage.total_retrieved - coverage.unique_sources,
        saturation=coverage.saturation,
        screening_truncated=coverage.screening_truncated,
        extraction_truncated=coverage.extraction_truncated,
        multi_source_themes=multi_source_themes,
        single_source_themes=single_source_themes,
        multi_source_problems=tiers[SupportTier.MULTI_SOURCE],
        tentative_problems=tiers[SupportTier.TENTATIVE],
        single_source_limitation_problems=tiers[
            SupportTier.SINGLE_SOURCE_LIMITATION
        ],
        contradicted_problems=tiers[SupportTier.CONTRADICTED],
    )
    reasons = _apply_rules(metrics, inventory.problems, thresholds)
    return MapAdequacyAssessment(
        run_id=run_id,
        brief_id=brief_id,
        field_map_id=field_map.id,
        inventory_id=inventory.id,
        status=(
            AdequacyStatus.INSUFFICIENT_COVERAGE
            if reasons
            else AdequacyStatus.ADEQUATE_FOR_IDEA_GENERATION
        ),
        reasons=reasons,
        thresholds=thresholds,
        metrics=metrics,
        problem_support=problem_support,
    )


def _apply_rules(
    metrics: AdequacyMetrics,
    problems: Sequence[ProblemEntry],
    thresholds: AdequacyThresholds,
) -> tuple[AdequacyReason, ...]:
    """Every rule, in fixed order, all of them evaluated — a consumer
    sees the complete diagnosis, not the first symptom."""
    reasons: list[AdequacyReason] = []
    budget_note = (
        " (screening or extraction was budget-truncated: a larger budget "
        "might change this)"
        if metrics.screening_truncated or metrics.extraction_truncated
        else ""
    )
    if metrics.relevant_sources < thresholds.min_relevant_sources:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.TOO_FEW_RELEVANT,
                f"{metrics.relevant_sources} relevant source(s) of "
                f"{metrics.screened} screened; the bar is "
                f"{thresholds.min_relevant_sources}{budget_note}",
            )
        )
    if metrics.grounded_sources < thresholds.min_grounded_sources:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.TOO_FEW_GROUNDED,
                f"{metrics.grounded_sources} source(s) with sufficient "
                f"accessible support ({metrics.metadata_only_relevant} "
                f"relevant source(s) were metadata-only); the bar is "
                f"{thresholds.min_grounded_sources}{budget_note}",
            )
        )
    if (
        len(metrics.families_with_relevant)
        < thresholds.min_families_with_relevant
    ):
        named = ", ".join(metrics.families_with_relevant) or "none"
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.FAMILY_COVERAGE_THIN,
                f"only {len(metrics.families_with_relevant)} query "
                f"famil(ies) produced relevant sources ({named}); the bar "
                f"is {thresholds.min_families_with_relevant}",
            )
        )
    if metrics.recent_grounded < thresholds.min_recent_grounded:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.RECENT_COVERAGE_THIN,
                f"{metrics.recent_grounded} grounded recent source(s); "
                f"the bar is {thresholds.min_recent_grounded}",
            )
        )
    if metrics.foundational_grounded < thresholds.min_foundational_grounded:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.FOUNDATIONAL_COVERAGE_THIN,
                f"{metrics.foundational_grounded} grounded foundational "
                f"source(s); the bar is "
                f"{thresholds.min_foundational_grounded}",
            )
        )
    if metrics.uncertain_fraction > thresholds.max_uncertain_fraction:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.EXCESSIVE_UNCERTAINTY,
                f"{metrics.uncertain_sources} of {metrics.screened} "
                f"screened source(s) were undecidable "
                f"({metrics.uncertain_fraction}); the bar is "
                f"{thresholds.max_uncertain_fraction}",
            )
        )
    if metrics.multi_source_themes < thresholds.min_multi_source_themes:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.THEME_SUPPORT_THIN,
                f"{metrics.multi_source_themes} theme(s) rest on two or "
                f"more distinct sources "
                f"({metrics.single_source_themes} single-source); the bar "
                f"is {thresholds.min_multi_source_themes}",
            )
        )
    cross_paper = (
        metrics.multi_source_problems + metrics.contradicted_problems
    )
    if cross_paper < thresholds.min_multi_source_problems:
        reasons.append(
            AdequacyReason(
                AdequacyReasonCode.PROBLEM_SUPPORT_THIN,
                f"{cross_paper} problem(s) rest on multiple distinct "
                f"sources; the bar is "
                f"{thresholds.min_multi_source_problems} "
                f"({metrics.tentative_problems} tentative, "
                f"{metrics.single_source_limitation_problems} "
                f"single-source limitation(s))",
            )
        )
    for problem in problems:
        distinct_total = len(
            set(problem.supporting_source_ids)
            | set(problem.conflicting_source_ids)
        )
        if problem.kind in _CROSS_PAPER_KINDS and distinct_total < 2:
            reasons.append(
                AdequacyReason(
                    AdequacyReasonCode.CROSS_PAPER_PATTERN_UNSUPPORTED,
                    f"problem {problem.statement!r} claims a cross-paper "
                    f"pattern ({problem.kind.value}) on {distinct_total} "
                    f"distinct source(s)",
                )
            )
    return tuple(reasons)


def require_adequate_for_idea_generation(
    store: MappingStore, assessment_id: str
) -> MapAdequacyAssessment:
    """The narrow Task 5C guard: load the durable assessment (tamper-
    checked by the store) and return it only when it says adequate.
    Anything else — unknown id, insufficient coverage — raises with the
    typed reasons. Task 5C itself does not exist yet; this is the door
    it will have to walk through."""
    assessment = store.get_adequacy(assessment_id)
    if assessment is None:
        raise InadequateFieldMapError(
            f"no adequacy assessment {assessment_id} is recorded; idea "
            f"generation requires an assessed field map"
        )
    if assessment.status is not AdequacyStatus.ADEQUATE_FOR_IDEA_GENERATION:
        raise InadequateFieldMapError(
            f"field map of run {assessment.run_id} is not adequate for "
            f"idea generation: "
            + "; ".join(
                f"{r.code.value}: {r.detail}" for r in assessment.reasons
            ),
            reasons=assessment.reasons,
        )
    return assessment
