"""Event-triggered critic escalation: review is earned, not scheduled.

An ordinary experiment goes ``execute -> deterministic checks -> director``
with no critique in between: the mechanical prediction test and the Tier-0
validation report already say everything a routine result can say. A critic
is invoked only when a result is *consequential* — and "consequential" is
decided by this deterministic, inspectable trigger, never by a model.

Conditions (each check is a few lines of arithmetic over the record):

* **contradictory replications** — conclusive tests of the same prediction
  disagree;
* **standing challenged** — a new conclusive test disagrees with the current
  epistemic assessment of its hypothesis;
* **unexpectedly large effect** — the observed value is far beyond the
  pre-registered threshold's own scale;
* **implementation uncertainty** — the run completed but deterministic
  validation found problems;
* **repeated failures** — the same intent has now failed several times;
* **explicit director request** — the director may always ask.

"About to become a major claim / branch decision" arrives through the last
condition: it is a judgment about intent, so the entity holding the intent
(the director) states it explicitly rather than having it inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.experiment import ExperimentResult
from ..core.prediction import Consistency, PredictionTest
from ..core.state import ResearchState
from ..runtime.validation import ValidationReport

_CHALLENGES = {
    # current settled verdict -> the test consistency that challenges it
    "supported": Consistency.INCONSISTENT,
    "refuted": Consistency.CONSISTENT,
}


@dataclass(frozen=True, slots=True)
class CriticTrigger:
    """Deterministic rules for when a result deserves a critic. All
    thresholds are visible fields, so the trigger is tunable and its ablation
    is one config edit."""

    large_effect_factor: float = 2.0
    """A conclusive observation counts as an unexpectedly large effect when
    it misses the pre-registered threshold by more than this multiple of the
    threshold's own scale."""

    relative_scale_floor: float = 0.1
    """The threshold's scale is ``max(tolerance, floor * |threshold|)`` — the
    floor keeps near-zero thresholds from flagging every observation."""

    repeated_failure_threshold: int = 2

    def reasons(
        self,
        state: ResearchState,
        *,
        result: ExperimentResult,
        validation: ValidationReport,
        test: PredictionTest | None,
        director_request: str | None = None,
    ) -> tuple[str, ...]:
        """Why this result deserves critique — empty means it does not."""
        reasons: list[str] = []

        if test is not None:
            reasons.extend(self._test_reasons(state, test))

        if result.succeeded and not validation.passed:
            failed = ", ".join(check.name for check in validation.failures)
            reasons.append(
                f"implementation uncertainty: run completed but deterministic "
                f"validation failed ({failed})"
            )

        failures = sum(
            1
            for attempt in state.attempts
            if not attempt.succeeded
            and attempt.status.is_terminal
            and attempt.action.targets
            and result.spec_id in attempt.action.targets
        )
        if failures >= self.repeated_failure_threshold:
            reasons.append(
                f"repeated failures: {failures} failed attempt(s) against "
                f"experiment {result.spec_id}"
            )

        if director_request:
            reasons.append(f"director request: {director_request}")

        return tuple(reasons)

    def _test_reasons(
        self, state: ResearchState, test: PredictionTest
    ) -> list[str]:
        reasons: list[str] = []
        tests = state.tests_for(test.prediction_id)
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
