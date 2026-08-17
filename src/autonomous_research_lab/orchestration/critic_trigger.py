"""Event-triggered critic escalation: review is earned, not scheduled.

An ordinary experiment goes ``execute -> deterministic checks -> director``
with no critique in between: the pre-commit validation gate and the
mechanical prediction test already say everything a routine result can say.
A critic is invoked only when a *scientifically valid* result is
consequential — and "consequential" is decided by this deterministic,
inspectable trigger, never by a model.

The conditions are scientific, exclusively:

* **contradictory replications** — conclusive tests of the same prediction
  disagree;
* **standing challenged** — a new conclusive test disagrees with the current
  settled epistemic assessment of its hypothesis;
* **unexpectedly large effect** — the observed value is far beyond the
  pre-registered threshold's own scale;
* **explicit director request** — the director may always ask.

What is deliberately *not* here: engineering trouble. A result that fails
deterministic validation never reaches this trigger (the gate rejects it
before commit), and repeated execution failures produce a deterministic
runtime note for the director — a debugging signal, not a question a critic
could answer. Asking an LLM whether a metric is missing would be paying for
an opinion about arithmetic.

Nor is *unverified* science here: under an admissibility policy (the
``admissible`` callback), a result that is not scientifically admissible
raises no scientific trigger reasons at all — its mechanically conclusive
test is a recorded observation, not a contradiction, a challenge, or a
large-effect finding — and inadmissible tests do not count toward
contradiction tallies. Implementation-invalid results have their own
response (the bounded repair path), which is not scientific critique.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.prediction import Consistency, PredictionTest
from ..core.state import ResearchState
from ..runtime.frontier import AdmissibilityCheck

_CHALLENGES = {
    # current settled verdict -> the test consistency that challenges it
    "supported": Consistency.INCONSISTENT,
    "refuted": Consistency.CONSISTENT,
}


def _admits(admissible: Callable[[str], bool] | None, result_id: str) -> bool:
    return admissible is None or admissible(result_id)


@dataclass(frozen=True, slots=True)
class CriticTrigger:
    """Deterministic rules for when a valid result deserves a critic. All
    thresholds are visible fields, so the trigger is tunable and its ablation
    is one config edit."""

    large_effect_factor: float = 2.0
    """A conclusive observation counts as an unexpectedly large effect when
    it misses the pre-registered threshold by more than this multiple of the
    threshold's own scale."""

    relative_scale_floor: float = 0.1
    """The threshold's scale is ``max(tolerance, floor * |threshold|)`` — the
    floor keeps near-zero thresholds from flagging every observation."""

    def reasons(
        self,
        state: ResearchState,
        *,
        test: PredictionTest | None,
        director_request: str | None = None,
        admissible: AdmissibilityCheck | None = None,
    ) -> tuple[str, ...]:
        """Why this result deserves critique — empty means it does not.

        A test whose result is not scientifically admissible produces no
        scientific reasons: an unverified observation cannot contradict,
        challenge standing, or count as a surprising effect. The explicit
        director request is the one reason that survives regardless.
        """
        reasons: list[str] = []
        if test is not None and _admits(admissible, test.result_id):
            reasons.extend(self._test_reasons(state, test, admissible))
        if director_request:
            reasons.append(f"director request: {director_request}")
        return tuple(reasons)

    def _test_reasons(
        self,
        state: ResearchState,
        test: PredictionTest,
        admissible: AdmissibilityCheck | None,
    ) -> list[str]:
        reasons: list[str] = []
        tests = tuple(
            t
            for t in state.tests_for(test.prediction_id)
            if _admits(admissible, t.result_id)
        )
        consistent = sum(
            1 for t in tests if t.consistency is Consistency.CONSISTENT
        )
        inconsistent = sum(
            1 for t in tests if t.consistency is Consistency.INCONSISTENT
        )
        if consistent and inconsistent:
            reasons.append(
                f"contradictory replications: {consistent} consistent vs "
                f"{inconsistent} inconsistent test(s) of prediction "
                f"{test.prediction_id}"
            )

        prediction = state.prediction(test.prediction_id)
        if prediction is None or test.observed is None:
            return reasons

        assessment = state.current_assessment(prediction.hypothesis_id)
        if (
            assessment is not None
            and _CHALLENGES.get(assessment.verdict.value) is test.consistency
        ):
            reasons.append(
                f"standing challenged: hypothesis currently assessed "
                f"{assessment.verdict}, new test is {test.consistency}"
            )

        scale = max(
            prediction.tolerance,
            self.relative_scale_floor * abs(prediction.threshold),
        )
        if scale > 0 and (
            abs(test.observed - prediction.threshold)
            > self.large_effect_factor * scale
        ):
            reasons.append(
                f"unexpectedly large effect: observed {test.observed} vs "
                f"threshold {prediction.threshold} (scale {scale:g})"
            )
        return reasons
