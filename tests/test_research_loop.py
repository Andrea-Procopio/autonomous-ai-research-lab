"""The end-to-end architectural contract.

state -> director -> spec -> executor -> result -> evidence -> updated state.

These assertions are about the shape of the pipeline, not about coin flips. If
one fails, a boundary has moved.
"""

from __future__ import annotations

from pathlib import Path

from autonomous_research_lab.core.actions import ResearchActionType
from autonomous_research_lab.core.claim import ClaimStatus, EvidenceRelation
from autonomous_research_lab.core.evidence import EvidenceKind
from autonomous_research_lab.core.experiment import ExperimentStatus
from autonomous_research_lab.core.hypothesis import HypothesisStatus
from autonomous_research_lab.knowledge.graph import ClaimEvidenceGraph
from examples.minimal_loop import run_minimal_loop


def test_loop_runs_to_completion(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)

    types = [action.action_type for action in outcome.state.history]
    assert types == [
        ResearchActionType.GENERATE_HYPOTHESIS,
        ResearchActionType.DESIGN_EXPERIMENT,
        ResearchActionType.RUN_EXPERIMENT,
        ResearchActionType.ANALYZE,
        ResearchActionType.SYNTHESIZE_FINDING,
        ResearchActionType.STOP_INVESTIGATION,
    ]
    assert outcome.halt_reason == "no open scientific work remains in this state"


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


def test_a_falsified_hypothesis_is_recorded_as_such(tmp_path: Path) -> None:
    """The demo hypothesis is false, and the system must say so rather than
    finding a reading of the data that keeps it alive."""
    outcome = run_minimal_loop(tmp_path)

    (hypothesis,) = outcome.state.hypotheses
    assert hypothesis.status is HypothesisStatus.FALSIFIED

    (evidence,) = outcome.store.evidence()
    assert evidence.kind is EvidenceKind.NULL_RESULT

    (claim,) = outcome.state.claims
    assert claim.status is ClaimStatus.REFUTED

    (link,) = outcome.state.evidence_links
    assert link.relation is EvidenceRelation.CONTRADICTS

    graph = ClaimEvidenceGraph.from_state(outcome.state, outcome.store)
    assert [s.claim.id for s in graph.contradicted()] == [claim.id]


def test_state_carries_its_own_lineage(tmp_path: Path) -> None:
    outcome = run_minimal_loop(tmp_path)
    assert outcome.state.parent_id is not None
    assert outcome.state.id != outcome.state.parent_id


def test_budget_is_charged_as_the_program_runs(tmp_path: Path) -> None:
    from examples.minimal_loop import initial_state

    start = initial_state()
    outcome = run_minimal_loop(tmp_path)

    assert outcome.state.budget.model_tokens < start.budget.model_tokens
    assert outcome.state.budget.wall_clock_seconds < start.budget.wall_clock_seconds


def test_loop_is_reproducible(tmp_path: Path) -> None:
    """Same inputs, same trajectory: deterministic ids make two runs comparable
    object by object, which is what makes a trajectory analysable later."""
    first = run_minimal_loop(tmp_path / "a")
    second = run_minimal_loop(tmp_path / "b")

    assert [a.id for a in first.state.history] == [a.id for a in second.state.history]
    assert [h.id for h in first.state.hypotheses] == [
        h.id for h in second.state.hypotheses
    ]
    assert first.store.evidence()[0].metrics == second.store.evidence()[0].metrics
