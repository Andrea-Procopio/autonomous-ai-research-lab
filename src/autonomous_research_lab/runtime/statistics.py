"""Exact inference over one prediction's replication family.

The statistician's arithmetic, and nothing else: pure functions from
immutable prediction tests to a verdict and the figures behind it. No
model authors a number here, and no number leaves without its method —
the rendered summary becomes the assessment's rationale, so every
figure rides inside a content-addressed, supersedable record that a
reviewer can challenge.

The test is deliberately the humblest exact one: a one-sided sign test.
Each conclusive prediction test is one Bernoulli observation — the
pre-registered comparison either held or it did not — and the p-value
is the exact binomial tail in the direction of the unanimous outcome,
``P[X >= k]`` under ``Binomial(n, 1/2)`` via ``math.comb``. At the
sample sizes this lab runs, anything more parametric would be
apparatus, not power; confidence intervals and power analysis are
recorded roadmap work, not silent omissions.

Multiple comparisons are disclosed, not hidden: the caller states how
many prediction families the hypothesis is being tested across, and the
threshold is Bonferroni-adjusted. The adjustment is pinned at
assessment time — sequential-analysis behavior an assessment's record
states rather than disguises.
"""

from __future__ import annotations

import math
import statistics as _stdlib_statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ..core.assessment import AssessmentVerdict
from ..core.prediction import Comparator, Consistency, Prediction, PredictionTest

DEFAULT_ALPHA: Final = 0.05


class SignDirection(StrEnum):
    """Which tail the exact test was taken in."""

    SUPPORTING = "supporting"
    CONTRARY = "contrary"
    NONE = "none"
    """No unanimous direction — a mixed or empty family tests neither."""


@dataclass(frozen=True, slots=True)
class FamilyStatistics:
    """The figures behind one verdict, all of them from the record."""

    metric: str
    comparator: Comparator
    threshold: float
    n: int
    values: tuple[float, ...]
    """The observed values, in the order the tests were given (the
    caller orders by seed where seeds are known)."""

    mean: float | None
    stdev: float | None
    """Sample standard deviation; ``None`` below two observations,
    because a spread of one number is not a measurement."""

    consistent_count: int
    direction: SignDirection
    p_value: float | None
    """Exact one-sided sign-test tail in ``direction``; ``None`` when
    the family has no unanimous direction to test."""

    effect: float | None
    """Mean minus threshold, signed so that positive favors the
    prediction; ``None`` for ``APPROXIMATELY``, whose 'effect' is a
    distance, not a direction."""

    alpha: float
    comparisons: int
    alpha_adjusted: float

    def render(self) -> str:
        """The stable one-line summary an assessment carries as its
        rationale — the whole computation, stated."""
        mean = "n/a" if self.mean is None else f"{self.mean:.6g}"
        stdev = "n/a" if self.stdev is None else f"{self.stdev:.6g}"
        effect = "n/a" if self.effect is None else f"{self.effect:+.6g}"
        p_value = "n/a" if self.p_value is None else f"{self.p_value:.6g}"
        return (
            f"exact sign test over the replication family: n={self.n}, "
            f"consistent {self.consistent_count}/{self.n}, "
            f"mean {mean} (stdev {stdev}) vs "
            f"{self.comparator.value} {self.threshold!r} "
            f"(effect {effect}), direction {self.direction.value}, "
            f"one-sided p={p_value}, "
            f"alpha {self.alpha:g} Bonferroni-adjusted to "
            f"{self.alpha_adjusted:.6g} across {self.comparisons} "
            f"comparison(s)"
        )


def assess_family(
    prediction: Prediction,
    tests: Sequence[PredictionTest],
    *,
    alpha: float = DEFAULT_ALPHA,
    comparisons: int = 1,
) -> tuple[AssessmentVerdict, FamilyStatistics]:
    """The verdict one prediction's conclusive tests support, exactly.

    ============================  =====================================
    family                        verdict
    ============================  =====================================
    no conclusive tests           UNDETERMINED — nothing was measured
    mixed outcomes                CONTESTED — the record disagrees with
                                  itself, and no tail is one-sided
    all consistent, p <= alpha'   SUPPORTED
    all consistent, p > alpha'    PLAUSIBLE — consistent so far, and
                                  underpowered to say more
    all contrary, p <= alpha'     REFUTED
    all contrary, p > alpha'      UNDETERMINED — consistently contrary,
                                  and underpowered to say even that
    ============================  =====================================

    where alpha' is the Bonferroni-adjusted threshold. The p-value is
    the tail in the direction of the unanimous outcome — the naive
    "P[X >= consistent] always" makes refutation unreachable, since an
    all-contrary family has zero consistent tests and a tail of one.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if comparisons < 1:
        raise ValueError("comparisons counts this family; at least 1")
    conclusive = [
        test
        for test in tests
        if test.consistency is not Consistency.INCONCLUSIVE
    ]
    values = tuple(
        test.observed for test in conclusive if test.observed is not None
    )
    n = len(conclusive)
    consistent = sum(
        1
        for test in conclusive
        if test.consistency is Consistency.CONSISTENT
    )
    adjusted = alpha / comparisons

    mean = _stdlib_statistics.fmean(values) if values else None
    stdev = (
        _stdlib_statistics.stdev(values) if len(values) >= 2 else None
    )
    effect = _effect(prediction, mean)

    if n == 0:
        direction, p_value = SignDirection.NONE, None
        verdict = AssessmentVerdict.UNDETERMINED
    elif 0 < consistent < n:
        direction, p_value = SignDirection.NONE, None
        verdict = AssessmentVerdict.CONTESTED
    elif consistent == n:
        direction = SignDirection.SUPPORTING
        p_value = _sign_tail(n, consistent)
        verdict = (
            AssessmentVerdict.SUPPORTED
            if p_value <= adjusted
            else AssessmentVerdict.PLAUSIBLE
        )
    else:
        direction = SignDirection.CONTRARY
        p_value = _sign_tail(n, n - consistent)
        verdict = (
            AssessmentVerdict.REFUTED
            if p_value <= adjusted
            else AssessmentVerdict.UNDETERMINED
        )

    return verdict, FamilyStatistics(
        metric=prediction.metric,
        comparator=prediction.comparator,
        threshold=prediction.threshold,
        n=n,
        values=values,
        mean=mean,
        stdev=stdev,
        consistent_count=consistent,
        direction=direction,
        p_value=p_value,
        effect=effect,
        alpha=alpha,
        comparisons=comparisons,
        alpha_adjusted=adjusted,
    )


def _sign_tail(n: int, successes: int) -> float:
    """Exact one-sided binomial tail: P[X >= successes] at p = 1/2."""
    tail = sum(math.comb(n, k) for k in range(successes, n + 1))
    return tail / float(2**n)


def _effect(prediction: Prediction, mean: float | None) -> float | None:
    """Mean minus threshold, signed so positive favors the prediction.

    Total over the comparator enum: ``APPROXIMATELY`` yields ``None``,
    because its 'effect' is a distance from the threshold, not a
    direction past it, and reporting a signed number would claim a
    direction nobody pre-registered.
    """
    if mean is None:
        return None
    if prediction.comparator in (
        Comparator.GREATER_THAN,
        Comparator.GREATER_OR_EQUAL,
    ):
        return mean - prediction.threshold
    if prediction.comparator in (
        Comparator.LESS_THAN,
        Comparator.LESS_OR_EQUAL,
    ):
        return prediction.threshold - mean
    return None
