"""Storing a commit bundle before it is applied.

The property that matters: a bundle read back from disk commits to the
same successor state as the bundle that was written. Everything else
here — content addressing, tamper loudness, storing facts by reference —
is in service of that one round trip surviving a dead process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptStatus,
)
from autonomous_research_lab.core.budget import ResourceCost
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.commit import CommitBundle
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    QuestionProposal,
    ResultProposal,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
from autonomous_research_lab.orchestration.transitions import commit_bundle
from autonomous_research_lab.persistence.commit_store import (
    BundleConflictError,
    BundleError,
    CommitBundleStore,
    bundle_id_of,
)

HYPOTHESIS = Hypothesis(statement="X causes Y.")
QUESTION = ResearchQuestion(text="Does X cause Y?")
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="under the standard setup",
    metric="effect_size",
    comparator=Comparator.GREATER_THAN,
    threshold=0.5,
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="compare the two arms",
    procedure="run both arms and compare",
    metrics=("effect_size",),
)
RESULT = ExperimentResult(
    spec_id=SPEC.id,
    job_id="job_1",
    status=ExperimentStatus.COMPLETED,
    command=("python", "run.py"),
    environment=Environment(python_version="3.11.0", platform="test"),
    metrics={"effect_size": 0.9},
)
EVIDENCE = Evidence(
    result_id=RESULT.id,
    spec_id=SPEC.id,
    kind=EvidenceKind.MEASUREMENT,
    observation="the effect size was 0.9",
    metrics={"effect_size": 0.9},
)
CLAIM = Claim(statement="X causes Y in this setup.", hypothesis_id=HYPOTHESIS.id)


def facts() -> InMemoryEvidenceStore:
    store = InMemoryEvidenceStore()
    store.record_result(RESULT)
    store.record_evidence(EVIDENCE)
    return store


def full_bundle(attempt_id: str = "att_1") -> CommitBundle:
    """One of every proposal kind, so the codec is exercised whole."""
    return CommitBundle(
        attempt_id=attempt_id,
        outcome=ActionOutcome(
            status=AttemptStatus.SUCCEEDED,
            produced=(
                QUESTION.id,
                HYPOTHESIS.id,
                PREDICTION.id,
                SPEC.id,
                RESULT.id,
                EVIDENCE.id,
                CLAIM.id,
            ),
            actual_cost=ResourceCost(usd=1.5, model_tokens=42),
        ),
        proposals=(
            QuestionProposal(QUESTION, proposer="scientist", motivation="why"),
            HypothesisProposal(HYPOTHESIS, proposer="scientist"),
            PredictionProposal(PREDICTION, proposer="scientist"),
            ExperimentProposal(SPEC, proposer="scientist"),
            ResultProposal(RESULT, proposer="executor"),
            EvidenceProposal(EVIDENCE, proposer="analyst"),
            ClaimProposal(
                CLAIM,
                links=(
                    EvidenceLink(
                        claim_id=CLAIM.id,
                        evidence_id=EVIDENCE.id,
                        relation=EvidenceRelation.SUPPORTS,
                    ),
                ),
                proposer="analyst",
            ),
            AssessmentProposal(
                EpistemicAssessment(
                    subject_id=CLAIM.id,
                    verdict=AssessmentVerdict.SUPPORTED,
                    method="review",
                    evidence_ids=(EVIDENCE.id,),
                ),
                proposer="critic",
            ),
        ),
    )


class TestRoundTrip:
    def test_a_stored_bundle_reads_back_identical(self, tmp_path: Path) -> None:
        store = CommitBundleStore(tmp_path)
        bundle = full_bundle()

        bundle_id = store.record(bundle)

        assert store.load(bundle_id, facts=facts()) == bundle

    def test_the_id_is_the_content(self, tmp_path: Path) -> None:
        store = CommitBundleStore(tmp_path)

        assert store.record(full_bundle()) == bundle_id_of(full_bundle())

    def test_storing_the_same_bundle_twice_writes_once(
        self, tmp_path: Path
    ) -> None:
        store = CommitBundleStore(tmp_path)

        first = store.record(full_bundle())
        second = store.record(full_bundle())

        assert first == second
        assert store.bundle_ids() == (first,)

    def test_two_attempts_produce_two_bundles(self, tmp_path: Path) -> None:
        store = CommitBundleStore(tmp_path)

        store.record(full_bundle("att_1"))
        store.record(full_bundle("att_2"))

        assert len(store.bundle_ids()) == 2

    def test_a_failed_bundle_round_trips_too(self, tmp_path: Path) -> None:
        store = CommitBundleStore(tmp_path)
        bundle = CommitBundle(
            attempt_id="att_1",
            outcome=ActionOutcome(status=AttemptStatus.FAILED, error="boom"),
        )

        bundle_id = store.record(bundle)

        assert store.load(bundle_id, facts=facts()) == bundle


class TestApplyingItAgain:
    def test_a_reloaded_bundle_commits_to_the_same_successor(
        self, tmp_path: Path
    ) -> None:
        """The whole point: a process that never ran the step can finish
        it, and arrive at the state the dead process would have."""
        action = ResearchAction(
            action_type=ResearchActionType.RUN_EXPERIMENT, rationale="r"
        )
        attempt = ActionAttempt(action=action).started()
        origin = ResearchState(objective="o").begin_attempt(attempt)
        bundle = full_bundle(attempt.id)

        direct = commit_bundle(origin, bundle, facts())

        store = CommitBundleStore(tmp_path)
        bundle_id = store.record(bundle)
        reloaded = store.load(bundle_id, facts=facts())
        recovered = commit_bundle(origin, reloaded, facts())

        assert recovered.id == direct.id


class TestFactsAreStoredByReference:
    def test_the_result_payload_is_not_copied_into_the_bundle(
        self, tmp_path: Path
    ) -> None:
        store = CommitBundleStore(tmp_path)
        bundle_id = store.record(full_bundle())

        text = (store.directory / f"{bundle_id}.json").read_text()
        payload = json.loads(text)

        result_entry = next(
            entry for entry in payload["proposals"] if entry["kind"] == "result"
        )
        assert result_entry == {
            "kind": "result",
            "result_id": RESULT.id,
            "proposer": "executor",
        }
        # None of the result's own content is duplicated here.
        assert "job_1" not in text
        assert "3.11.0" not in text

    def test_a_bundle_whose_facts_are_gone_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """Rather than reconstructing a smaller bundle than the one that
        was written."""
        store = CommitBundleStore(tmp_path)
        bundle_id = store.record(full_bundle())

        with pytest.raises(BundleError, match=RESULT.id):
            store.load(bundle_id, facts=InMemoryEvidenceStore())


class TestTamperLoudness:
    def test_an_edited_bundle_fails_to_load(self, tmp_path: Path) -> None:
        store = CommitBundleStore(tmp_path)
        bundle_id = store.record(full_bundle())
        path = store.directory / f"{bundle_id}.json"
        payload = json.loads(path.read_text())
        payload["attempt_id"] = "att_somewhere_else"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        with pytest.raises(BundleError, match="re-derives"):
            store.load(bundle_id, facts=facts())

    def test_a_missing_bundle_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match="no bundle"):
            CommitBundleStore(tmp_path).load("bun_nowhere", facts=facts())

    def test_malformed_json_is_not_silently_empty(self, tmp_path: Path) -> None:
        store = CommitBundleStore(tmp_path)
        (store.directory / "bun_broken.json").write_text("{not json")

        with pytest.raises(BundleError, match="not valid JSON"):
            store.load("bun_broken", facts=facts())

    def test_two_different_bundles_cannot_share_one_name(
        self, tmp_path: Path
    ) -> None:
        store = CommitBundleStore(tmp_path)
        bundle_id = store.record(full_bundle())
        path = store.directory / f"{bundle_id}.json"
        payload = json.loads(path.read_text())
        payload["attempt_id"] = "att_2"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        with pytest.raises(BundleConflictError, match="never rewritten"):
            store.record(full_bundle())
