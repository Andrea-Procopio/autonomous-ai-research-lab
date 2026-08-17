"""The atomic commit boundary.

The invariant under test: **a successful action cannot claim outputs that do
not exist in the resulting state/store** — and a rejected bundle changes
nothing at all.
"""

from __future__ import annotations

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptStatus,
)
from autonomous_research_lab.core.claim import Claim
from autonomous_research_lab.core.commit import CommitBundle
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.proposals import (
    ClaimProposal,
    HypothesisProposal,
    PredictionProposal,
)
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.orchestration.transitions import (
    TransitionError,
    commit_bundle,
)

HYPOTHESIS = Hypothesis(statement="X causes Y.")
ACTION = ResearchAction(
    action_type=ResearchActionType.GENERATE_HYPOTHESIS, rationale="r"
)


def begun() -> tuple[ResearchState, ActionAttempt]:
    attempt = ActionAttempt(action=ACTION).started()
    state = ResearchState(objective="o").begin_attempt(attempt)
    return state, attempt


def succeeded(*produced: str) -> ActionOutcome:
    return ActionOutcome(status=AttemptStatus.SUCCEEDED, produced=produced)


class TestBundleShape:
    def test_a_failed_bundle_cannot_carry_proposals(self) -> None:
        with pytest.raises(ValueError, match="commits nothing"):
            CommitBundle(
                attempt_id="att_1",
                outcome=ActionOutcome(status=AttemptStatus.FAILED, error="boom"),
                proposals=(HypothesisProposal(HYPOTHESIS, proposer="t"),),
            )

    def test_a_failed_bundle_cannot_claim_produced_ids(self) -> None:
        with pytest.raises(ValueError, match="produced"):
            CommitBundle(
                attempt_id="att_1",
                outcome=ActionOutcome(
                    status=AttemptStatus.FAILED,
                    error="boom",
                    produced=("hyp_ghost",),
                ),
            )


class TestAtomicCommit:
    def test_success_with_valid_produced_objects(self) -> None:
        state, attempt = begun()
        bundle = CommitBundle(
            attempt_id=attempt.id,
            outcome=succeeded(HYPOTHESIS.id),
            proposals=(HypothesisProposal(HYPOTHESIS, proposer="t"),),
        )
        after = commit_bundle(state, bundle, InMemoryEvidenceStore())

        assert after.hypothesis(HYPOTHESIS.id) is not None
        resolved = next(a for a in after.attempts if a.id == attempt.id)
        assert resolved.succeeded
        assert resolved.outcome is not None
        assert resolved.outcome.produced == (HYPOTHESIS.id,)

    def test_success_claiming_a_missing_output_is_rejected(self) -> None:
        state, attempt = begun()
        bundle = CommitBundle(
            attempt_id=attempt.id,
            outcome=succeeded(HYPOTHESIS.id, "ev_never_created"),
            proposals=(HypothesisProposal(HYPOTHESIS, proposer="t"),),
        )
        with pytest.raises(TransitionError, match="ev_never_created"):
            commit_bundle(state, bundle, InMemoryEvidenceStore())

    def test_failure_attempt_with_no_outputs_is_valid(self) -> None:
        state, attempt = begun()
        bundle = CommitBundle(
            attempt_id=attempt.id,
            outcome=ActionOutcome(status=AttemptStatus.FAILED, error="model refused"),
        )
        after = commit_bundle(state, bundle, InMemoryEvidenceStore())

        resolved = next(a for a in after.attempts if a.id == attempt.id)
        assert resolved.status is AttemptStatus.FAILED
        assert after.hypotheses == ()

    def test_rejected_bundle_leaves_the_original_state_unchanged(self) -> None:
        """First proposal valid, second orphaned: the whole transition is
        rejected and nothing — not even the valid half — reaches the state."""
        state, attempt = begun()
        orphan = Prediction(
            hypothesis_id="hyp_unknown",
            condition="c",
            metric="m",
            comparator=Comparator.GREATER_THAN,
            threshold=0.5,
        )
        bundle = CommitBundle(
            attempt_id=attempt.id,
            outcome=succeeded(HYPOTHESIS.id, orphan.id),
            proposals=(
                HypothesisProposal(HYPOTHESIS, proposer="t"),
                PredictionProposal(orphan, proposer="t"),
            ),
        )
        before_id = state.id
        with pytest.raises(TransitionError, match="unknown hypothesis"):
            commit_bundle(state, bundle, InMemoryEvidenceStore())

        assert state.id == before_id
        assert state.hypotheses == ()
        running = next(a for a in state.attempts if a.id == attempt.id)
        assert not running.status.is_terminal  # still open for a retry

    def test_unknown_attempt_is_rejected(self) -> None:
        state = ResearchState(objective="o")
        bundle = CommitBundle(attempt_id="att_ghost", outcome=succeeded())
        with pytest.raises(TransitionError, match="never begun"):
            commit_bundle(state, bundle, InMemoryEvidenceStore())

    def test_an_attempt_resolves_at_most_once(self) -> None:
        state, attempt = begun()
        store = InMemoryEvidenceStore()
        first = CommitBundle(attempt_id=attempt.id, outcome=succeeded())
        after = commit_bundle(state, first, store)
        with pytest.raises(TransitionError, match="already terminal"):
            commit_bundle(after, first, store)

    def test_claim_links_count_as_committed_outputs(self) -> None:
        """payload_ids covers everything a proposal brings in — a claim
        bundle may legitimately claim its link ids as produced."""
        state, attempt = begun()
        state = state.upsert_hypothesis(HYPOTHESIS)
        claim = Claim(statement="s", hypothesis_id=HYPOTHESIS.id)
        bundle = CommitBundle(
            attempt_id=attempt.id,
            outcome=succeeded(claim.id),
            proposals=(ClaimProposal(claim, proposer="t"),),
        )
        after = commit_bundle(state, bundle, InMemoryEvidenceStore())
        assert after.claim(claim.id) is not None
