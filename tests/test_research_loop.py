"""The end-to-end architectural contract.

state -> director -> attempt -> proposals -> transition layer -> state',
with results from the executor, predictions checked mechanically, and standing
settled by explicit assessment.

These assertions are about the shape of the pipeline, not about coin flips. If
one fails, a boundary has moved.
"""

from __future__ import annotations

from pathlib import Path

from autonomous_research_lab.core.actions import ResearchActionType
from autonomous_research_lab.core.assessment import AssessmentVerdict
from autonomous_research_lab.core.claim import EvidenceRelation
from autonomous_research_lab.core.evidence import EvidenceKind
from autonomous_research_lab.core.experiment import ExperimentStatus
from autonomous_research_lab.core.hypothesis import HypothesisStatus
from autonomous_research_lab.core.prediction import PredictionStatus
from autonomous_research_lab.knowledge.graph import ClaimEvidenceGraph
from examples.minimal_loop import initial_state, run_minimal_loop

EXPECTED_TRAJECTORY = [
    ResearchActionType.GENERATE_HYPOTHESIS,
    ResearchActionType.DERIVE_PREDICTION,
    ResearchActionType.DESIGN_EXPERIMENT,
    ResearchActionType.RUN_EXPERIMENT,
    ResearchActionType.ANALYZE,
    ResearchActionType.SYNTHESIZE_FINDING,
    ResearchActionType.ASSESS_CLAIM,
    ResearchActionType.STOP_INVESTIGATION,
]


def test_loop_runs_to_completion(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)

    assert [a.action_type for a in outcome.state.history] == EXPECTED_TRAJECTORY
    assert outcome.halt_reason == "no open scientific work remains in this state"


def test_every_step_is_an_attempt_with_a_succeeded_outcome(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)
    attempts = outcome.state.attempts

    # One attempt per non-stop action, all terminal, all succeeded, each
    # naming what it produced.
    assert len(attempts) == len(EXPECTED_TRAJECTORY) - 1
    for attempt in attempts:
        assert attempt.succeeded
        assert attempt.outcome is not None
        assert attempt.outcome.produced


def test_metrics_originate_from_a_process_that_ran(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)
    (result,) = outcome.store.results()

    assert result.status is ExperimentStatus.COMPLETED
    assert result.exit_code == 0
    assert result.command[-1].endswith("coin_bias.py")
    assert set(result.metrics) >= {"heads_rate", "n_draws"}
    # Provenance sufficient to attempt a re-run.
    assert result.seed is not None
    assert result.environment.python_version
    assert result.runtime_seconds > 0.0
    assert any(Path(log).exists() for log in result.logs)


def test_evidence_quotes_the_result_it_came_from(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)
    (result,) = outcome.store.results()
    (evidence,) = outcome.store.evidence()

    assert evidence.result_id == result.id
    assert evidence.metrics["heads_rate"] == result.metrics["heads_rate"]
    assert evidence.id in outcome.state.evidence_ids


def test_falsification_flows_through_the_full_chain(tmp_path: Path) -> None:
    """The demo hypothesis is false. The system must say so at every layer —
    mechanically at the prediction, factually at the evidence link, and
    epistemically at the assessment — rather than finding a reading of the
    data that keeps it alive."""
    outcome = run_minimal_loop(tmp_path)
    state = outcome.state

    (prediction,) = state.predictions
    assert prediction.status is PredictionStatus.FAILED

    (evidence,) = outcome.store.evidence()
    assert evidence.kind is EvidenceKind.NULL_RESULT

    (link,) = state.evidence_links
    assert link.relation is EvidenceRelation.CONTRADICTS

    (claim,) = state.claims
    assessment = state.current_assessment(claim.id)
    assert assessment is not None
    assert assessment.verdict is AssessmentVerdict.REFUTED
    assert assessment.method == "demo:prediction-check-v0"
    assert evidence.id in assessment.evidence_ids

    (hypothesis,) = state.hypotheses
    assert hypothesis.status is HypothesisStatus.FALSIFIED

    graph = ClaimEvidenceGraph.from_state(state, outcome.store)
    assert [c.claim.id for c in graph.contradicted()] == [claim.id]
    assert graph.unassessed() == ()


def test_state_carries_its_own_lineage(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)
    assert outcome.state.parent_id is not None
    assert outcome.state.id != outcome.state.parent_id


def test_budget_is_charged_as_the_program_runs(tmp_path: Path) -> None:
    start = initial_state()
    outcome = run_minimal_loop(tmp_path)

    assert outcome.state.budget.model_tokens < start.budget.model_tokens
    assert outcome.state.budget.wall_clock_seconds < start.budget.wall_clock_seconds


def test_semantic_objects_reproduce_and_events_do_not(tmp_path: Path) -> None:
    """Two identical runs agree on every purely semantic object — hypotheses,
    predictions, specs, the actions over them, measured values — which is what
    makes trajectories comparable. Events (attempts, executions) are distinct
    occurrences, and objects downstream of an execution (results, evidence,
    actions targeting them) inherit that event identity: the two runs really
    did perform different executions, and the ids say so."""
    first = run_minimal_loop(tmp_path / "a")
    second = run_minimal_loop(tmp_path / "b")

    # Semantic prefix — everything up to the run — is identical across runs.
    semantic = [
        ResearchActionType.GENERATE_HYPOTHESIS,
        ResearchActionType.DERIVE_PREDICTION,
        ResearchActionType.DESIGN_EXPERIMENT,
        ResearchActionType.RUN_EXPERIMENT,  # targets the spec: content-addressed
    ]
    first_ids = [a.id for a in first.state.history[: len(semantic)]]
    second_ids = [a.id for a in second.state.history[: len(semantic)]]
    assert first_ids == second_ids
    assert [h.id for h in first.state.hypotheses] == [
        h.id for h in second.state.hypotheses
    ]
    assert [p.id for p in first.state.predictions] == [
        p.id for p in second.state.predictions
    ]
    assert first.store.evidence()[0].metrics == second.store.evidence()[0].metrics

    # Event layer: distinct occurrences, run to run.
    first_attempts = {a.id for a in first.state.attempts}
    second_attempts = {a.id for a in second.state.attempts}
    assert first_attempts.isdisjoint(second_attempts)
    assert first.store.results()[0].id != second.store.results()[0].id
