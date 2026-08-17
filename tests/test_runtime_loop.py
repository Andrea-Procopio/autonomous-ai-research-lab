"""The runtime's end-to-end invariants: sparse calls, triggered critique,
recorded metrics, removable components."""

from __future__ import annotations

from pathlib import Path

from autonomous_research_lab.core.actions import ResearchActionType
from autonomous_research_lab.core.assessment import AssessmentVerdict
from autonomous_research_lab.evidence.validation import (
    ChainIssueKind,
    validate_evidence_chain,
)
from autonomous_research_lab.orchestration.director import (
    Deliberation,
    RuleBasedFrontierDirector,
)
from autonomous_research_lab.orchestration.synthesis import SynthesisReview
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.escalation import ReasoningTier
from autonomous_research_lab.runtime.frontier import ResearchFrontier
from autonomous_research_lab.runtime.metrics import JsonlRuntimeMetrics
from autonomous_research_lab.runtime.playbook import PlaybookAdvice
from examples.runtime_loop import (
    ESCALATED_SEEDS,
    ESCALATED_THRESHOLD,
    DemoRun,
    experiment_report,
    run_runtime_loop,
)

EXPECTED_NORMAL_TRAJECTORY = [
    ResearchActionType.GENERATE_HYPOTHESIS,
    ResearchActionType.DERIVE_PREDICTION,
    ResearchActionType.DESIGN_EXPERIMENT,
    ResearchActionType.RUN_EXPERIMENT,
    ResearchActionType.SYNTHESIZE_FINDING,
    ResearchActionType.ASSESS_CLAIM,
    ResearchActionType.STOP_INVESTIGATION,
]


def _escalated(tmp_path: Path, config: RuntimeConfig | None = None) -> DemoRun:
    return run_runtime_loop(
        tmp_path,
        seeds=ESCALATED_SEEDS,
        threshold=ESCALATED_THRESHOLD,
        config=config,
    )


class CountingDirector:
    """RuleBasedFrontierDirector with an invocation counter, to pin the
    fast-path property: one deliberation per step, nothing else."""

    def __init__(self) -> None:
        self.inner = RuleBasedFrontierDirector()
        self.deliberations = 0
        self.syntheses = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def deliberate(
        self,
        frontier: ResearchFrontier,
        *,
        advice: PlaybookAdvice | None = None,
        tier: ReasoningTier = ReasoningTier.ROUTINE,
        max_candidates: int = 3,
    ) -> Deliberation:
        self.deliberations += 1
        return self.inner.deliberate(
            frontier, advice=advice, tier=tier, max_candidates=max_candidates
        )

    def synthesize(
        self,
        frontier: ResearchFrontier,
        *,
        tier: ReasoningTier = ReasoningTier.STRONG,
    ) -> SynthesisReview:
        self.syntheses += 1
        return self.inner.synthesize(frontier, tier=tier)


def test_normal_run_walks_the_deliverable_trajectory(tmp_path: Path) -> None:
    run = run_runtime_loop(tmp_path)
    state = run.outcome.state

    assert [
        a.action_type for a in state.history
    ] == EXPECTED_NORMAL_TRAJECTORY
    assert run.outcome.halt_reason == (
        "no open scientific work remains on the frontier"
    )
    (hypothesis,) = state.hypotheses
    assessment = state.current_assessment(hypothesis.id)
    assert assessment is not None
    assert assessment.verdict is AssessmentVerdict.REFUTED


def test_fast_path_needs_no_separate_evaluator_calls(tmp_path: Path) -> None:
    """One director invocation per step performs generation, valuation and
    selection — and the intermediate candidate set is still logged."""
    from autonomous_research_lab.evidence.store import InMemoryEvidenceStore
    from autonomous_research_lab.execution.local import LocalExecutor
    from autonomous_research_lab.orchestration.loop import ResearchRuntime
    from autonomous_research_lab.roles.base import RoleName
    from autonomous_research_lab.runtime.playbook import EmpiricalMLPlaybook
    from examples.runtime_loop import (
        DemoCritic,
        DemoEngineer,
        DemoScientist,
        initial_state,
    )

    director = CountingDirector()
    runtime = ResearchRuntime(
        config=RuntimeConfig(),
        director=director,
        roles={
            RoleName.RESEARCH_DIRECTOR: DemoScientist(threshold=0.55, seeds=(7,)),
            RoleName.RESEARCH_ENGINEER: DemoEngineer(
                LocalExecutor(tmp_path / "runs")
            ),
            RoleName.RESULT_ANALYST: DemoCritic(),
        },
        store=InMemoryEvidenceStore(),
        playbook=EmpiricalMLPlaybook(),
    )
    outcome = runtime.run(initial_state())

    # Exactly one deliberation per step; the only other director-seat calls
    # are explicit slow-loop syntheses.
    assert director.deliberations == len(outcome.reports)
    for report in outcome.reports:
        record = report.record
        assert record.evaluated  # the candidate set was preserved
        # One invocation did all three jobs, and the trajectory says so.
        assert record.generator == record.evaluator == record.policy
        assert record.generator == director.name


def test_the_deliberation_carries_a_small_candidate_set(tmp_path: Path) -> None:
    run = _escalated(tmp_path)
    replicate = experiment_report(run, ResearchActionType.REPLICATE)

    candidates = replicate.deliberation.candidates
    assert 2 <= len(candidates) <= 3
    assert replicate.deliberation.reasoning  # pairwise rationale preserved
    assert "prefer" in replicate.deliberation.reasoning


def test_ordinary_experiment_iteration_costs_two_calls(tmp_path: Path) -> None:
    """The performance invariant: director 1, executor 1, critic 0."""
    run = run_runtime_loop(tmp_path)
    report = experiment_report(run, ResearchActionType.RUN_EXPERIMENT)

    assert report.llm_calls == 2
    assert not report.critic_invoked
    assert report.critic_reasons == ()
    assert report.validation is not None
    assert report.validation.passed


def test_consequential_iteration_adds_exactly_the_critic(tmp_path: Path) -> None:
    """Contradictory replication: director 1, executor 1, critic 1 — plus
    the slow loop, which the contradiction also wakes."""
    run = _escalated(tmp_path)
    report = experiment_report(run, ResearchActionType.REPLICATE)

    assert report.critic_invoked
    assert any("contradictory replications" in r for r in report.critic_reasons)
    synthesis_calls = 1 if report.synthesis else 0
    assert report.llm_calls == 3 + synthesis_calls

    # The critic's judgment is on the record, grounded in evidence.
    state = run.outcome.state
    (hypothesis,) = state.hypotheses
    assessment = state.current_assessment(hypothesis.id)
    assert assessment is not None
    assert assessment.verdict is AssessmentVerdict.CONTESTED
    assert assessment.evidence_ids


def test_disabling_the_critic_removes_the_call_but_keeps_the_data(
    tmp_path: Path,
) -> None:
    run = _escalated(tmp_path, config=RuntimeConfig(critic_enabled=False))
    report = experiment_report(run, ResearchActionType.REPLICATE)

    assert not report.critic_invoked
    # The trigger still fired and was recorded — the ablation needs the data.
    assert any("contradictory replications" in r for r in report.critic_reasons)


def test_disabling_synthesis_leaves_only_the_fast_loop(tmp_path: Path) -> None:
    run = _escalated(tmp_path, config=RuntimeConfig(synthesis_enabled=False))
    assert all(report.synthesis is None for report in run.outcome.reports)


def test_synthesis_wakes_on_contradiction_and_before_stopping(
    tmp_path: Path,
) -> None:
    run = _escalated(tmp_path)
    replicate = experiment_report(run, ResearchActionType.REPLICATE)
    assert replicate.synthesis is not None  # the contradiction woke it
    final = run.outcome.reports[-1]
    assert final.synthesis is not None  # and so did stopping


def test_playbook_is_optional_and_the_loop_still_terminates(
    tmp_path: Path,
) -> None:
    run = _escalated(tmp_path, config=RuntimeConfig(playbook_enabled=False))
    assert run.outcome.halt_reason == (
        "no open scientific work remains on the frontier"
    )


def test_runtime_metrics_are_recorded_per_decision(tmp_path: Path) -> None:
    run = run_runtime_loop(tmp_path)
    metrics = JsonlRuntimeMetrics(run.metrics_path)
    records = metrics.read()

    assert len(records) == len(run.outcome.reports)
    decision_ids = {r.record.id for r in run.outcome.reports}
    for record in records:
        assert record["decision_id"] in decision_ids
        assert isinstance(record["llm_calls"], int)
        assert record["llm_calls"] >= 1
        assert record["action_type"]
        assert record["rationale"]  # the raw decision rationale is preserved
        assert "reasoning_tier" in record
        assert "critic_invoked" in record
        assert "experiment_seconds" in record


def test_the_committed_chain_survives_the_chain_validator(
    tmp_path: Path,
) -> None:
    normal = run_runtime_loop(tmp_path / "normal")
    assert validate_evidence_chain(normal.outcome.state, normal.store) == ()

    escalated = _escalated(tmp_path / "escalated")
    issues = validate_evidence_chain(escalated.outcome.state, escalated.store)
    # The escalated run's only issues are the genuine contradictions.
    assert issues != ()
    assert {i.kind for i in issues} == {ChainIssueKind.CONTRADICTION}


def test_every_decision_boundary_state_reconstructs(tmp_path: Path) -> None:
    run = run_runtime_loop(tmp_path)
    for report in run.outcome.reports:
        for state_id in (
            report.record.state_before_id,
            report.record.state_after_id,
        ):
            assert state_id is not None
            assert run.states.load(state_id).id == state_id
