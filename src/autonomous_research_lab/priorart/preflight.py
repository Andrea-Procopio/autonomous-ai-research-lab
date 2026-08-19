"""The budget-coherence preflight: a challenge that cannot complete the
work its own directive permits is refused before the first call.

The Task 5D.1 live evidence motivating this module: the directive
allowed six families at five results each — up to thirty fresh works
plus the cited injection — against a screening cap of twenty, so a
*successful* retrieval mechanically truncated (candidate 2: 23 pooled,
20 screened, 3 truncated) and the shortfall was then reported as a
scientific deficiency. Separately, the default call budget covered the
worst case of exactly three candidates while the ideation layer defaults
to five, and nothing detected the mismatch before network execution.

:func:`check_budget_coherence` runs after the portfolio door and before
any model or network call. It verifies, from the directive, the loaded
candidates, and the challenger's own wiring:

* **retrieval fits screening** — the worst-case pool (families times
  results per query, plus the largest cited injection) cannot exceed
  the screening cap, so ``SCREENING_TRUNCATED`` can never be the
  mechanical product of a successful retrieval;
* **comparison is reachable** — the comparison cap covers the
  threshold's minimum compared works;
* **the threshold is reachable** — a pool that clears
  ``min_unique_sources`` must be screenable without truncation, or
  ``DISTINGUISHED`` is unreachable by construction;
* **calls fit** — the worst-case gated calls (query proposal, abstract
  and metadata screening batches, comparison, each with its bounded
  corrective call, per candidate) fit ``max_model_calls``.

Every violated inequality is collected and named together, so one
refusal carries the complete diagnosis. The runtime
:class:`~.challenger.PriorArtBudgetError` guard stays in place as
defense in depth, but after a passing preflight the worst-case bound
makes it unreachable from a normal run. Budget exhaustion becomes an
exceptional execution outcome, never the expected result of a directive
that passed preflight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from ..ideation.records import CandidateIdea
from .assessment import PriorArtThresholds
from .directive import PriorArtDirective
from .records import PriorArtQueryFamily


class PriorArtPreflightError(RuntimeError):
    """The directive and wiring cannot complete the work the directive
    itself permits. Raised before any model or network call, naming
    every violated inequality."""


@dataclass(frozen=True, slots=True)
class PriorArtCallPlan:
    """The worst-case arithmetic of one challenge run, computed by
    trusted code from the directive, the loaded portfolio, and the
    challenger's wiring. Diagnostic: printed by drivers, asserted by
    the preflight, recorded nowhere."""

    candidates: int
    families: int
    max_cited: int
    worst_pool_per_candidate: int
    screening_capacity: int
    worst_screening_calls_per_candidate: int
    worst_calls_per_candidate: int
    worst_calls_total: int


def check_budget_coherence(
    *,
    directive: PriorArtDirective,
    candidates: Sequence[CandidateIdea],
    thresholds: PriorArtThresholds,
    screening_batch_size: int,
    max_corrective_calls: int,
) -> PriorArtCallPlan:
    """Verify the directive can complete its own maximum work; return
    the computed call plan, or raise naming every violation.

    The screening-call bound uses ``ceil(S/b) + 1``: abstract-level and
    metadata-only sources screen in separate batches, and the worst
    split of ``S`` sources across the two adds at most one extra batch
    over screening them together."""
    families = len(PriorArtQueryFamily)
    max_cited = max(
        (len(candidate.cited_source_ids) for candidate in candidates),
        default=0,
    )
    worst_pool = families * directive.results_per_query + max_cited
    screening_calls = (
        ceil(directive.max_screened_per_candidate / screening_batch_size) + 1
    )
    per_candidate = (2 + screening_calls) * (1 + max_corrective_calls)
    total = len(candidates) * per_candidate
    plan = PriorArtCallPlan(
        candidates=len(candidates),
        families=families,
        max_cited=max_cited,
        worst_pool_per_candidate=worst_pool,
        screening_capacity=directive.max_screened_per_candidate,
        worst_screening_calls_per_candidate=screening_calls,
        worst_calls_per_candidate=per_candidate,
        worst_calls_total=total,
    )
    violations: list[str] = []
    if worst_pool > directive.max_screened_per_candidate:
        violations.append(
            f"retrieval exceeds the screening cap: {families} families at "
            f"{directive.results_per_query} results plus up to "
            f"{max_cited} cited sources can pool {worst_pool} works "
            f"against max_screened_per_candidate="
            f"{directive.max_screened_per_candidate}, so a successful "
            f"retrieval would mechanically truncate"
        )
    if directive.max_compared_works < thresholds.min_compared_works:
        violations.append(
            f"comparison is unreachable: max_compared_works="
            f"{directive.max_compared_works} cannot satisfy the "
            f"threshold's min_compared_works="
            f"{thresholds.min_compared_works}"
        )
    if directive.max_screened_per_candidate < thresholds.min_unique_sources:
        violations.append(
            f"the source threshold is unreachable: a pool clearing "
            f"min_unique_sources={thresholds.min_unique_sources} cannot "
            f"be screened without truncation under "
            f"max_screened_per_candidate="
            f"{directive.max_screened_per_candidate}"
        )
    if total > directive.max_model_calls:
        violations.append(
            f"the model-call budget cannot cover the promised work: "
            f"{len(candidates)} candidates x {2 + screening_calls} gated "
            f"stages x {1 + max_corrective_calls} calls each is "
            f"{total} worst-case calls against max_model_calls="
            f"{directive.max_model_calls}"
        )
    if violations:
        raise PriorArtPreflightError(
            "the directive cannot complete the work it permits: "
            + "; ".join(violations)
        )
    return plan
