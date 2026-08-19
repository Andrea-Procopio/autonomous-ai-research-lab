"""The deterministic prior-art verdict, and the one door into a
challenge run.

:func:`assess_prior_art` is trusted code all the way down, the
:func:`~..mapping.adequacy.assess_adequacy` discipline applied to
falsification: the metrics are kept apart from the conclusion, the
thresholds travel inside the assessment so a reloaded verdict carries
its own bar, every rule is evaluated (never short-circuited) so the
record holds the complete diagnosis, and the aggregation fails closed —
``NOVELTY_UNRESOLVED`` is the default that coverage must argue a
candidate out of, never a disappointment to retry.

The verdict semantics are deliberately narrow. ``OVERLAPPING`` needs one
accepted comparison whose substantial match the gate already forced to
ground itself dimension by dimension in a source's accessible text.
``DISTINGUISHED`` means only that the candidate is materially
differentiated from the closest works *this bounded search surfaced*,
under adequate recorded coverage — it is never proof of novelty, and no
value of this enum certifies anything about the world's literature. A
metadata-only source blocks ``DISTINGUISHED`` exactly when it was
screened as a *material* potential overlap — a decision the gate
refuses to record without an attested hypothesis naming the candidate
claim at risk and the source text supporting the concern. A
metadata-only source screened undecidable does not block: with no
abstract, undecidability restates the access level the coverage
already counts, and the Task 5D.1 live evidence showed that treating
it as ambiguity made every realistic pool block on access alone —
never on an overlap signal. The refusal stays honest because the
counts stay on the record, and deciding a material ambiguity "is the
same overlap" as some compared work would still be a model judgment
where only trusted code may conclude.

:func:`require_candidates_for_prior_art` is the door, mirroring
``require_adequate_for_idea_generation`` one stage down: a challenge run
enters through a durable ideation run record whose portfolio actually
holds candidates, or it does not start.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum

from ..core.ids import content_id
from ..ideation.records import IdeationRunRecord
from ..ideation.store import IdeationStore
from .records import (
    PriorArtCoverage,
    PriorArtQueryFamily,
    PriorArtScreeningRecord,
    SimilarityDecision,
    SimilarityLabel,
    WorkComparison,
)


class PriorArtVerdict(StrEnum):
    """What this bounded challenge concluded about one candidate.
    ``DISTINGUISHED`` describes this corpus only — absence from it is
    never proof of novelty."""

    OVERLAPPING = "overlapping"
    DISTINGUISHED = "distinguished"
    NOVELTY_UNRESOLVED = "novelty_unresolved"


class PriorArtReasonCode(StrEnum):
    """Typed grounds on which a challenge declines to distinguish a
    candidate. Every fired reason is recorded, whatever the verdict."""

    FAMILY_COVERAGE_INCOMPLETE = "family_coverage_incomplete"
    TOO_FEW_UNIQUE_SOURCES = "too_few_unique_sources"
    EXCESSIVE_UNCERTAINTY = "excessive_uncertainty"
    METADATA_AMBIGUITY = "metadata_ambiguity"
    NO_COMPARABLE_WORK = "no_comparable_work"
    UNCOMPARED_POTENTIAL_OVERLAP = "uncompared_potential_overlap"
    SCREENING_TRUNCATED = "screening_truncated"


@dataclass(frozen=True, slots=True)
class PriorArtReason:
    code: PriorArtReasonCode
    detail: str

    def __post_init__(self) -> None:
        if not self.detail.strip():
            raise ValueError("a reason names its specifics")


@dataclass(frozen=True, slots=True)
class PriorArtThresholds:
    """The recorded bar a DISTINGUISHED verdict must clear. Travels
    inside every assessment: a reloaded verdict carries the thresholds
    it was judged against.

    The bases are part of the semantics, fixed by the assessing code:
    ``min_unique_sources`` is measured against the in-cutoff screenable
    pool (unique sources minus post-cutoff exclusions) — a work the
    cutoff excludes can never be screened, so it cannot help ground
    differentiation — and ``max_undecidable_fraction`` bounds
    undecidable screens among abstract-level screens only, because a
    metadata-only source is expected to screen undecidable: that
    restates its access level, already counted by the coverage, and
    billing it here again would be the same missing abstract twice.
    Both operands of each basis travel in every coverage record, so any
    historical verdict can be re-derived under either reading."""

    min_unique_sources: int = 10
    max_undecidable_fraction: float = 0.34
    min_compared_works: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("min_unique_sources", self.min_unique_sources),
            ("min_compared_works", self.min_compared_works),
        ):
            if value < 1:
                raise ValueError(f"{label} must be at least 1")
        if not 0.0 < self.max_undecidable_fraction <= 1.0:
            raise ValueError(
                "max_undecidable_fraction must be a fraction in (0, 1]"
            )


@dataclass(frozen=True, slots=True)
class PriorArtAssessment:
    """One candidate's verdict, with the coverage, thresholds, and typed
    reasons that produced it. The verdict-shape invariants are
    structural: an OVERLAPPING verdict without a grounded overlapping
    work, or a DISTINGUISHED verdict carrying unresolved reasons, cannot
    be recorded."""

    run_id: str
    candidate_id: str
    directive_id: str
    verdict: PriorArtVerdict
    overlapping_work_ids: tuple[str, ...]
    compared_work_ids: tuple[str, ...]
    reasons: tuple[PriorArtReason, ...]
    thresholds: PriorArtThresholds
    coverage: PriorArtCoverage
    id: str = field(default="")

    def __post_init__(self) -> None:
        if (self.verdict is PriorArtVerdict.OVERLAPPING) != bool(
            self.overlapping_work_ids
        ):
            raise ValueError(
                "an OVERLAPPING verdict and a grounded overlapping work "
                "imply each other"
            )
        if self.verdict is PriorArtVerdict.DISTINGUISHED and self.reasons:
            raise ValueError(
                "a DISTINGUISHED verdict cannot carry unresolved reasons"
            )
        if (
            self.verdict is PriorArtVerdict.NOVELTY_UNRESOLVED
            and not self.reasons
        ):
            raise ValueError(
                "an unresolved verdict names why it is unresolved"
            )
        if not set(self.overlapping_work_ids) <= set(self.compared_work_ids):
            raise ValueError(
                "an overlapping work must be among the compared works"
            )
        if len(set(self.compared_work_ids)) != len(self.compared_work_ids):
            raise ValueError("compared works are listed once each")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "paa",
                    self.run_id,
                    self.candidate_id,
                    self.directive_id,
                    self.verdict,
                    self.overlapping_work_ids,
                    self.compared_work_ids,
                    tuple((r.code, r.detail) for r in self.reasons),
                    tuple(
                        getattr(self.thresholds, entry.name)
                        for entry in fields(PriorArtThresholds)
                    ),
                    tuple(
                        getattr(self.coverage, entry.name)
                        for entry in fields(PriorArtCoverage)
                    ),
                ),
            )


def assess_prior_art(
    *,
    run_id: str,
    candidate_id: str,
    directive_id: str,
    screenings: Sequence[PriorArtScreeningRecord],
    comparisons: Sequence[WorkComparison],
    coverage: PriorArtCoverage,
    metadata_source_ids: frozenset[str],
    thresholds: PriorArtThresholds,
) -> PriorArtAssessment:
    """Compute one candidate's verdict from the accepted records alone.

    ``metadata_source_ids`` is trusted code's account of which pooled
    sources carried no abstract. Every rule is evaluated; the reasons
    list is the complete diagnosis, recorded even when a grounded
    overlap settles the verdict on its own."""
    for record in screenings:
        if record.run_id != run_id or record.candidate_id != candidate_id:
            raise ValueError(
                "a screening from another run or candidate cannot enter "
                "this assessment"
            )
    for comparison in comparisons:
        if (
            comparison.run_id != run_id
            or comparison.candidate_id != candidate_id
        ):
            raise ValueError(
                "a comparison from another run or candidate cannot enter "
                "this assessment"
            )
    compared_ids = tuple(entry.source_id for entry in comparisons)
    if len(compared_ids) != coverage.compared_works:
        raise ValueError(
            "the coverage must count exactly the recorded comparisons"
        )
    ambiguous = tuple(
        record
        for record in screenings
        if record.source_id in metadata_source_ids
        and record.decision is SimilarityDecision.POTENTIAL_OVERLAP
    )
    if len(ambiguous) != coverage.metadata_ambiguous:
        raise ValueError(
            "the coverage must count exactly the material "
            "metadata-ambiguous screens"
        )

    reasons: list[PriorArtReason] = []
    executed = set(coverage.families_executed)
    required = {family.value for family in PriorArtQueryFamily}
    if executed != required:
        missing = ", ".join(sorted(required - executed))
        reasons.append(
            PriorArtReason(
                PriorArtReasonCode.FAMILY_COVERAGE_INCOMPLETE,
                f"the {missing} famil"
                f"{'ies were' if ',' in missing else 'y was'} never "
                f"searched; every family must run before differentiation "
                f"can be claimed",
            )
        )
    screenable = coverage.unique_sources - coverage.post_cutoff_excluded
    if screenable < thresholds.min_unique_sources:
        reasons.append(
            PriorArtReason(
                PriorArtReasonCode.TOO_FEW_UNIQUE_SOURCES,
                f"{screenable} in-cutoff screenable sources "
                f"({coverage.unique_sources} unique, "
                f"{coverage.post_cutoff_excluded} excluded post-cutoff, "
                f"{coverage.known_prior_art_listed} candidate-cited) "
                f"against a threshold of {thresholds.min_unique_sources}; "
                f"a pool this thin cannot ground differentiation",
            )
        )
    abstract_screens = tuple(
        record
        for record in screenings
        if record.source_id not in metadata_source_ids
    )
    if abstract_screens:
        undecided = sum(
            1
            for record in abstract_screens
            if record.decision is SimilarityDecision.UNDECIDABLE
        )
        fraction = round(undecided / len(abstract_screens), 4)
        if fraction > thresholds.max_undecidable_fraction:
            reasons.append(
                PriorArtReason(
                    PriorArtReasonCode.EXCESSIVE_UNCERTAINTY,
                    f"{undecided} of {len(abstract_screens)} "
                    f"abstract-level screens were undecidable "
                    f"({fraction}, threshold "
                    f"{thresholds.max_undecidable_fraction}); the pool is "
                    f"not understood well enough to distinguish against",
                )
            )
    if ambiguous:
        accounts = []
        for record in ambiguous:
            hypothesis = record.overlap_hypothesis
            if hypothesis is not None:
                accounts.append(
                    f"{record.source_id} (claim at risk: "
                    f"{hypothesis.candidate_claim!r}; dimension: "
                    f"{hypothesis.dimension.value})"
                )
            else:
                accounts.append(
                    f"{record.source_id} (recorded without an attested "
                    f"hypothesis; a pre-5D.2 screen blocks fail-closed)"
                )
        reasons.append(
            PriorArtReason(
                PriorArtReasonCode.METADATA_AMBIGUITY,
                f"metadata-only source"
                f"{'s' if len(ambiguous) > 1 else ''} screened as "
                f"materially overlapping with no abstract to compare: "
                f"{'; '.join(accounts)}",
            )
        )
    if coverage.compared_works < thresholds.min_compared_works:
        reasons.append(
            PriorArtReason(
                PriorArtReasonCode.NO_COMPARABLE_WORK,
                f"{coverage.compared_works} works compared against a "
                f"threshold of {thresholds.min_compared_works}; absence "
                f"of comparable work is never proof of novelty",
            )
        )
    uncompared = tuple(
        record.source_id
        for record in screenings
        if record.decision is SimilarityDecision.POTENTIAL_OVERLAP
        and record.source_id not in set(compared_ids)
        and record.source_id not in metadata_source_ids
    )
    if uncompared:
        reasons.append(
            PriorArtReason(
                PriorArtReasonCode.UNCOMPARED_POTENTIAL_OVERLAP,
                f"potentially overlapping work"
                f"{'s' if len(uncompared) > 1 else ''} "
                f"{', '.join(uncompared)} never reached comparison; the "
                f"budget truncated exactly the works most likely to "
                f"falsify",
            )
        )
    if coverage.screening_truncated:
        reasons.append(
            PriorArtReason(
                PriorArtReasonCode.SCREENING_TRUNCATED,
                f"{coverage.screening_truncated} pooled source"
                f"{'s were' if coverage.screening_truncated > 1 else ' was'}"
                f" never screened; unscreened possible prior art blocks "
                f"differentiation",
            )
        )

    overlapping = tuple(
        entry.source_id
        for entry in comparisons
        if entry.similarity is SimilarityLabel.SUBSTANTIAL_MATCH
    )
    if overlapping:
        verdict = PriorArtVerdict.OVERLAPPING
    elif reasons:
        verdict = PriorArtVerdict.NOVELTY_UNRESOLVED
    else:
        verdict = PriorArtVerdict.DISTINGUISHED
    return PriorArtAssessment(
        run_id=run_id,
        candidate_id=candidate_id,
        directive_id=directive_id,
        verdict=verdict,
        overlapping_work_ids=overlapping,
        compared_work_ids=compared_ids,
        reasons=tuple(reasons),
        thresholds=thresholds,
        coverage=coverage,
    )


class MissingCandidatePortfolioError(RuntimeError):
    """The one refusal the door raises: no durable run record, a refusal
    portfolio, or a candidate that fails to load."""


def require_candidates_for_prior_art(
    store: IdeationStore, run_record_id: str
) -> IdeationRunRecord:
    """The single entrance to a prior-art challenge: a durable ideation
    run record whose portfolio holds loadable candidates. An honest
    refusal run has nothing to challenge; an absent or partial record
    is refused before any model call."""
    record = store.get_run(run_record_id)
    if record is None:
        raise MissingCandidatePortfolioError(
            f"no ideation run record {run_record_id} in this store; a "
            f"challenge enters through a durable portfolio"
        )
    if not record.candidate_ids:
        raise MissingCandidatePortfolioError(
            f"ideation run {record.run_id} recorded an honest refusal; "
            f"there is no portfolio to challenge"
        )
    for candidate_id in record.candidate_ids:
        if store.get_idea(candidate_id) is None:
            raise MissingCandidatePortfolioError(
                f"candidate {candidate_id} named by run record "
                f"{run_record_id} is not in this store; refusing a "
                f"partial portfolio"
            )
    return record
