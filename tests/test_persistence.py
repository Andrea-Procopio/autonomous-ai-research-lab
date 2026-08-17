"""State snapshots: persist -> reload -> equivalent state.

The store is content-addressed: the snapshot filename is the state's content
id, identical states deduplicate, and loading recomputes the id from the
reconstructed content so a corrupt snapshot fails loudly instead of quietly
resurrecting a different state.
"""

from __future__ import annotations

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
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.experiment import (
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.persistence.state_store import (
    FileStateStore,
    SnapshotError,
    serialize_state,
)


def rich_state() -> ResearchState:
    """A state exercising every field, including nested attempts and tests."""
    question = ResearchQuestion(text="Is the stream fair?", importance="i")
    hypothesis = Hypothesis(
        statement="The stream is biased.",
        rationale="r",
        assumptions=("independent draws",),
        question_id=question.id,
    )
    prediction = Prediction(
        hypothesis_id=hypothesis.id,
        condition="4000 draws",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=0.55,
        expectation="rate >= 0.55",
    )
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="estimate rate",
        procedure="draw and count",
        metrics=("heads_rate",),
        seeds=(7,),
        estimated_cost=ResourceCost(wall_clock_seconds=10.0),
    )
    test = PredictionTest(
        prediction_id=prediction.id,
        result_id="res_1",
        metric="heads_rate",
        observed=0.51,
        consistency=Consistency.INCONSISTENT,
        detail="observed 0.51 vs ge 0.55",
    )
    claim = Claim(statement="Biased.", scope="seeded", hypothesis_id=hypothesis.id)
    link = EvidenceLink(
        claim_id=claim.id, evidence_id="ev_1", relation=EvidenceRelation.CONTRADICTS
    )
    assessment = EpistemicAssessment(
        subject_id=claim.id,
        verdict=AssessmentVerdict.REFUTED,
        method="test:v0",
        evidence_ids=("ev_1",),
        confidence=0.7,
    )
    action = ResearchAction(
        action_type=ResearchActionType.ANALYZE, rationale="r", targets=("res_1",)
    )
    # Occurrence ids pinned so the fixture is reproducible call to call —
    # exactly what reconstruction does when it reads stored attempt ids.
    done = ActionAttempt(action=action, id="att_fixture_1").started()
    failed = ActionAttempt(action=action, id="att_fixture_2").started()

    state = ResearchState(
        objective="o",
        questions=(question,),
        budget=ResearchBudget(wall_clock_seconds=100.0, usd=5.0, model_tokens=1000),
    )
    state = state.upsert_hypothesis(hypothesis)
    state = state.upsert_prediction(prediction)
    state = state.add_experiment(spec)
    state = state.record_result(
        ResultRef(result_id="res_1", spec_id=spec.id, status=ExperimentStatus.COMPLETED)
    )
    state = state.record_evidence("ev_1")
    state = state.record_prediction_test(test)
    state = state.upsert_claim(claim)
    state = state.link_evidence(link)
    state = state.record_assessment(assessment)
    state = state.begin_attempt(done)
    state = state.resolve_attempt(
        done.resolved(
            ActionOutcome(
                status=AttemptStatus.SUCCEEDED,
                produced=("ev_1",),
                actual_cost=ResourceCost(usd=0.1),
            )
        )
    )
    state = state.begin_attempt(failed)
    state = state.resolve_attempt(
        failed.resolved(ActionOutcome(status=AttemptStatus.FAILED, error="boom"))
    )
    state = state.apply(action)
    return state.charge(ResourceCost(usd=1.0))


def test_persist_then_reload_is_the_equivalent_state(tmp_path: Path) -> None:
    store = FileStateStore(tmp_path)
    state = rich_state()

    store.persist(state)
    reloaded = store.load(state.id)

    assert reloaded == state
    assert reloaded.id == state.id
    assert reloaded.parent_id == state.parent_id


def test_serialization_is_deterministic(tmp_path: Path) -> None:
    """Same content, same bytes — the property content addressing rests on."""
    assert serialize_state(rich_state()) == serialize_state(rich_state())


def test_identical_content_deduplicates(tmp_path: Path) -> None:
    store = FileStateStore(tmp_path)
    state = rich_state()
    first = store.persist(state)
    second = store.persist(rich_state())

    assert first == second
    assert store.state_ids() == (state.id,)


def test_lineage_chain_survives_persistence(tmp_path: Path) -> None:
    store = FileStateStore(tmp_path)
    parent = ResearchState(objective="o")
    child = parent.upsert_hypothesis(Hypothesis(statement="s"))
    store.persist(parent)
    store.persist(child)

    reloaded_child = store.load(child.id)
    assert reloaded_child.parent_id == parent.id
    assert store.load(parent.id) == parent


def test_unknown_state_id_raises(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="no snapshot"):
        FileStateStore(tmp_path).load("st_missing")


def test_tampered_snapshot_is_rejected(tmp_path: Path) -> None:
    """A snapshot whose content no longer hashes to its id must not load —
    that is the difference between reconstruction and fabrication."""
    store = FileStateStore(tmp_path)
    state = ResearchState(objective="o")
    path = store.persist(state)
    path.write_text(path.read_text().replace('"o"', '"tampered"'))

    with pytest.raises(SnapshotError, match="reconstructs to"):
        store.load(state.id)


def test_malformed_snapshot_is_rejected(tmp_path: Path) -> None:
    store = FileStateStore(tmp_path)
    state = ResearchState(objective="o")
    path = store.persist(state)
    path.write_text("{not json")
    with pytest.raises(SnapshotError, match="not valid JSON"):
        store.load(state.id)
