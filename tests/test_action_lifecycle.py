"""Action lifecycle semantics: intent -> attempt -> outcome.

The invariant under test: **a failed attempt never makes work look done.**
Completion is a property of succeeded outcomes; ``history`` is an audit trail
that nothing operational reads.
"""

from __future__ import annotations

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptStatus,
)
from autonomous_research_lab.core.experiment import ExperimentStatus, ResultRef
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.candidates import RuleBasedCandidateGenerator

ANALYZE = ResearchAction(
    action_type=ResearchActionType.ANALYZE,
    rationale="result res_1 has not been read into evidence",
    targets=("res_1",),
)


class TestAttemptInvariants:
    def test_terminal_outcome_only(self) -> None:
        with pytest.raises(ValueError, match="terminal"):
            ActionOutcome(status=AttemptStatus.RUNNING)

    def test_terminal_attempt_requires_outcome(self) -> None:
        with pytest.raises(ValueError, match="requires an outcome"):
            ActionAttempt(action=ANALYZE, status=AttemptStatus.FAILED)

    def test_status_and_outcome_must_agree(self) -> None:
        with pytest.raises(ValueError, match="disagrees"):
            ActionAttempt(
                action=ANALYZE,
                status=AttemptStatus.FAILED,
                outcome=ActionOutcome(status=AttemptStatus.SUCCEEDED),
            )

    def test_resolution_preserves_identity(self) -> None:
        attempt = ActionAttempt(action=ANALYZE).started()
        resolved = attempt.resolved(ActionOutcome(status=AttemptStatus.SUCCEEDED))
        assert resolved.id == attempt.id
        assert resolved.succeeded

    def test_a_terminal_attempt_cannot_be_resolved_again(self) -> None:
        attempt = ActionAttempt(action=ANALYZE).started()
        resolved = attempt.resolved(ActionOutcome(status=AttemptStatus.FAILED))
        with pytest.raises(ValueError, match="already terminal"):
            resolved.resolved(ActionOutcome(status=AttemptStatus.SUCCEEDED))

    def test_succeeded_outcomes_carry_no_error(self) -> None:
        with pytest.raises(ValueError, match="cannot carry an error"):
            ActionOutcome(status=AttemptStatus.SUCCEEDED, error="boom")


def state_with_unanalyzed_result() -> ResearchState:
    return ResearchState(objective="o").record_result(
        ResultRef(result_id="res_1", spec_id="exp_1", status=ExperimentStatus.COMPLETED)
    )


def offered_analyze(state: ResearchState) -> bool:
    return any(
        c.action.action_type is ResearchActionType.ANALYZE
        and "res_1" in c.action.targets
        for c in RuleBasedCandidateGenerator().generate(state)
    )


class TestFailureLeavesWorkOpen:
    """The scenario from the requirements, end to end:

    ANALYZE -> attempt 1 fails -> still unresolved -> attempt 2 succeeds
    -> resolved.
    """

    def test_retry_after_failure(self) -> None:
        state = state_with_unanalyzed_result()
        assert offered_analyze(state)

        # Attempt 1 fails.
        first = ActionAttempt(action=ANALYZE).started()
        state = state.begin_attempt(first)
        state = state.resolve_attempt(
            first.resolved(
                ActionOutcome(status=AttemptStatus.FAILED, error="parser crashed")
            )
        )

        # The failure is on the record, and the analysis is still open work.
        assert not state.has_succeeded(ResearchActionType.ANALYZE, "res_1")
        assert offered_analyze(state)

        # Attempt 2 — a new occurrence of the same intent — succeeds.
        second = ActionAttempt(action=ANALYZE).started()
        assert second.id != first.id
        assert second.action.id == first.action.id
        state = state.begin_attempt(second)
        state = state.resolve_attempt(
            second.resolved(
                ActionOutcome(status=AttemptStatus.SUCCEEDED, produced=("ev_1",))
            )
        )

        # Now, and only now, the work is done — with both attempts preserved.
        assert state.has_succeeded(ResearchActionType.ANALYZE, "res_1")
        assert not offered_analyze(state)
        assert len(state.attempts_for(ANALYZE.id)) == 2

    def test_in_flight_work_is_not_reoffered(self) -> None:
        state = state_with_unanalyzed_result()
        state = state.begin_attempt(ActionAttempt(action=ANALYZE).started())
        assert state.in_flight(ResearchActionType.ANALYZE, "res_1")
        assert not offered_analyze(state)

    def test_history_is_not_operational_truth(self) -> None:
        """An action in ``history`` proves nothing about completion: here the
        audit trail says ANALYZE happened, no attempt succeeded, and the work
        is still offered."""
        state = state_with_unanalyzed_result().apply(ANALYZE)
        assert ANALYZE in state.history
        assert not state.has_succeeded(ResearchActionType.ANALYZE, "res_1")
        assert offered_analyze(state)
