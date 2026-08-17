"""The research runtime, end to end, on the coin-bias question.

Two scenarios over the same wiring:

**Normal** (one seed, threshold 0.55) — the deliverable's ordinary loop::

    frontier -> director deliberates once -> deterministic routing
    -> executor runs in job-private isolation -> deterministic validation
    gate -> critic trigger evaluates to FALSE -> commit -> new frontier

The experiment iteration makes exactly two reasoning-seat invocations
(director, executor) — and, because every role here is rule-based, zero
actual model calls, which the metrics record as such. Analysis of the
ordinary result is deterministic transcription.

**Escalated** (two seeds straddling threshold 0.5) — a replication
contradicts the first run, the deterministic critic trigger fires, and the
critic reviews the contradiction as a third invocation. The synthesis slow
loop also runs, because a contradiction appeared.

The roles here are rule-based mocks standing where model-backed roles will
sit; like real roles they read only their invocation's context and return
proposals. Run with::

    python examples/runtime_loop.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.budget import ResearchBudget
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.experiment import ExperimentSpec
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
    PredictionTest,
)
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    ResultProposal,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import (
    EvidenceStore,
    InMemoryEvidenceStore,
)
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.orchestration.director import RuleBasedFrontierDirector
from autonomous_research_lab.orchestration.loop import (
    ResearchRuntime,
    RunOutcome,
    StepReport,
)
from autonomous_research_lab.orchestration.trajectory import JsonlTrajectoryLogger
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.metrics import JsonlRuntimeMetrics
from autonomous_research_lab.runtime.playbook import EmpiricalMLPlaybook

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPT = REPO_ROOT / "examples" / "experiments" / "coin_bias.py"

OBJECTIVE = "Determine whether the seeded draw stream used by our samplers is fair."
QUESTION = "Does the seeded Bernoulli stream deviate from a fair coin?"
N_DRAWS = 4000

#: Normal scenario: one seed, a threshold the data clearly misses.
NORMAL_SEEDS = (7,)
NORMAL_THRESHOLD = 0.55
#: Escalated scenario: threshold 0.5 with seeds whose observed rates straddle
#: it (seed 3 -> ~0.504, seed 0 -> ~0.492), so the replication contradicts
#: the first run and the critic trigger fires.
ESCALATED_SEEDS = (3, 0)
ESCALATED_THRESHOLD = 0.5


# -- mock roles ---------------------------------------------------------------
# Rule-based stand-ins for model-backed roles. Each reads only its
# invocation's context and returns proposals; none touches state.


class DemoScientist(ResearchRole):
    """The scientist seat: hypotheses, predictions, designs, syntheses."""

    def __init__(self, *, threshold: float, seeds: tuple[int, ...]) -> None:
        self._threshold = threshold
        self._seeds = seeds

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_DIRECTOR

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {
                ResearchActionType.GENERATE_HYPOTHESIS,
                ResearchActionType.DERIVE_PREDICTION,
                ResearchActionType.DESIGN_EXPERIMENT,
                ResearchActionType.SYNTHESIZE_FINDING,
            }
        )

    def suitability(
        self,
        state: ResearchState,  # noqa: ARG002 - demo roles fit everything
        action: ResearchAction,  # noqa: ARG002
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        match invocation.assignment.action_type:
            case ResearchActionType.GENERATE_HYPOTHESIS:
                return self._generate(invocation)
            case ResearchActionType.DERIVE_PREDICTION:
                return self._derive(invocation)
            case ResearchActionType.DESIGN_EXPERIMENT:
                return self._design(invocation)
            case ResearchActionType.SYNTHESIZE_FINDING:
                return self._synthesize(invocation)
            case _:  # pragma: no cover - contract enforced by the invocation
                raise NotImplementedError(invocation.assignment.action_type)

    def _generate(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        hypothesis = Hypothesis(
            statement=(
                f"The seeded Bernoulli stream is biased toward heads, with a "
                f"rate of at least {self._threshold:.2f}."
            ),
            rationale="Chosen so the demo exercises falsification.",
            assumptions=("Draws are independent.",),
            question_id=(
                invocation.assignment.targets[0]
                if invocation.assignment.targets
                else None
            ),
        )
        return (
            HypothesisProposal(hypothesis=hypothesis, proposer="demo:scientist"),
        )

    def _derive(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        (hypothesis,) = invocation.context.hypotheses
        prediction = Prediction(
            hypothesis_id=hypothesis.id,
            condition=f"{N_DRAWS} draws from the seeded generator",
            metric="heads_rate",
            comparator=Comparator.GREATER_OR_EQUAL,
            threshold=self._threshold,
            expectation=(
                f"The observed heads rate is at least {self._threshold:.2f}."
            ),
        )
        return (
            PredictionProposal(prediction=prediction, proposer="demo:scientist"),
        )

    def _design(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        (prediction,) = invocation.context.predictions
        spec = ExperimentSpec(
            prediction_id=prediction.id,
            objective="Estimate the heads rate of the seeded stream.",
            procedure=(
                f"Draw {N_DRAWS} Bernoulli samples from the seeded generator "
                f"and report the observed heads rate."
            ),
            metrics=("heads_rate", "n_draws", "abs_deviation_from_half"),
            baselines=("fair coin, rate 0.5",),
            seeds=self._seeds,
        )
        return (ExperimentProposal(spec=spec, proposer="demo:scientist"),)

    def _synthesize(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        (evidence,) = invocation.context.evidence
        (hypothesis,) = invocation.context.hypotheses
        (spec,) = invocation.context.experiments
        test = next(
            t
            for t in invocation.context.prediction_tests
            if t.result_id == evidence.result_id
        )
        claim = Claim(
            statement=hypothesis.statement,
            scope=spec.procedure,
            hypothesis_id=hypothesis.id,
        )
        relation = {
            Consistency.CONSISTENT: EvidenceRelation.SUPPORTS,
            Consistency.INCONSISTENT: EvidenceRelation.CONTRADICTS,
            Consistency.INCONCLUSIVE: EvidenceRelation.INCONCLUSIVE,
        }[test.consistency]
        link = EvidenceLink(
            claim_id=claim.id,
            evidence_id=evidence.id,
            relation=relation,
            rationale=f"pre-registered prediction tested {test.consistency}: "
            f"{test.detail}",
        )
        return (
            ClaimProposal(claim=claim, links=(link,), proposer="demo:scientist"),
        )


class DemoEngineer(ResearchRole):
    """The executor seat: short-lived, isolated, given a spec and its prior
    runs — nothing else."""

    def __init__(self, executor: LocalExecutor) -> None:
        self._executor = executor

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {ResearchActionType.RUN_EXPERIMENT, ResearchActionType.REPLICATE}
        )

    def suitability(
        self,
        state: ResearchState,  # noqa: ARG002 - demo roles fit everything
        action: ResearchAction,  # noqa: ARG002
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        (spec,) = invocation.context.experiments
        used = {r.seed for r in invocation.context.results}
        seed = next(s for s in spec.seeds if s not in used)
        job = ExperimentJob(
            spec_id=spec.id,
            command=(sys.executable, str(EXPERIMENT_SCRIPT)),
            config={"n_draws": N_DRAWS},
            seed=seed,
            timeout_seconds=120.0,
            required_artifacts=("metrics.json",),
        )
        job_id = self._executor.submit(job)
        result = self._executor.collect(job_id)
        return (ResultProposal(result=result, proposer="executor:local"),)


class DemoCritic(ResearchRole):
    """The critic/analyst seat: assessment on demand, review when triggered."""

    @property
    def name(self) -> RoleName:
        return RoleName.RESULT_ANALYST

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {ResearchActionType.ANALYZE, ResearchActionType.ASSESS_CLAIM}
        )

    def suitability(
        self,
        state: ResearchState,  # noqa: ARG002 - demo roles fit everything
        action: ResearchAction,  # noqa: ARG002
    ) -> RoleSuitability:
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        if invocation.assignment.action_type is ResearchActionType.ANALYZE:
            return self._review(invocation)
        return self._assess(invocation)

    def _verdict(
        self, tests: tuple[PredictionTest, ...]
    ) -> AssessmentVerdict:
        consistent = any(
            t.consistency is Consistency.CONSISTENT for t in tests
        )
        inconsistent = any(
            t.consistency is Consistency.INCONSISTENT for t in tests
        )
        if consistent and inconsistent:
            return AssessmentVerdict.CONTESTED
        if inconsistent:
            return AssessmentVerdict.REFUTED
        if consistent:
            return AssessmentVerdict.SUPPORTED
        return AssessmentVerdict.UNDETERMINED

    def _review(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        """The triggered path: review a consequential result and put a
        judgment about its hypothesis on the record."""
        (hypothesis,) = invocation.context.hypotheses
        verdict = self._verdict(invocation.context.prediction_tests)
        assessment = EpistemicAssessment(
            subject_id=hypothesis.id,
            verdict=verdict,
            method="demo-critic:triggered-review:v1",
            evidence_ids=tuple(e.id for e in invocation.context.evidence),
            scope=invocation.context.experiments[0].procedure
            if invocation.context.experiments
            else "",
            rationale="; ".join(invocation.context.notes),
        )
        return (
            AssessmentProposal(assessment=assessment, proposer="demo:critic"),
        )

    def _assess(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        (claim,) = invocation.context.claims
        verdict = self._verdict(invocation.context.prediction_tests)
        # Cite everything considered: the verdict weighs the full test
        # family, so the citation covers the full conclusive evidence too.
        evidence_ids = tuple(e.id for e in invocation.context.evidence)
        rationale = (
            "verdict follows the full set of mechanical prediction tests "
            "under their pre-registered conditions"
        )
        proposals: list[Proposal] = [
            AssessmentProposal(
                assessment=EpistemicAssessment(
                    subject_id=claim.id,
                    verdict=verdict,
                    method="demo-critic:assess:v1",
                    evidence_ids=evidence_ids,
                    scope=claim.scope,
                    rationale=rationale,
                ),
                proposer="demo:critic",
            )
        ]
        for hypothesis in invocation.context.hypotheses:
            proposals.append(
                AssessmentProposal(
                    assessment=EpistemicAssessment(
                        subject_id=hypothesis.id,
                        verdict=verdict,
                        method="demo-critic:assess:v1",
                        evidence_ids=evidence_ids,
                        rationale=rationale,
                    ),
                    proposer="demo:critic",
                )
            )
        return tuple(proposals)


# -- wiring -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DemoRun:
    outcome: RunOutcome
    store: EvidenceStore
    states: FileStateStore
    trajectory_path: Path
    metrics_path: Path


def initial_state() -> ResearchState:
    question = ResearchQuestion(
        text=QUESTION,
        importance=(
            "Every downstream experiment draws from this stream; an "
            "undetected bias would confound all of them."
        ),
    )
    return ResearchState(
        objective=OBJECTIVE,
        questions=(question,),
        budget=ResearchBudget(
            wall_clock_seconds=3600.0, gpu_hours=0.0, usd=10.0,
            model_tokens=200_000,
        ),
    )


def run_runtime_loop(
    run_root: Path | str | None = None,
    *,
    seeds: tuple[int, ...] = NORMAL_SEEDS,
    threshold: float = NORMAL_THRESHOLD,
    config: RuntimeConfig | None = None,
    max_steps: int = 24,
) -> DemoRun:
    root = Path(run_root) if run_root else Path(mkdtemp())
    # This demo predates the verification layer and wires none of it, so it
    # runs as the *explicitly* ablated lab: governance is switched off in
    # config rather than inferred from the absence of verification records.
    config = config or RuntimeConfig(verification_governance_enabled=False)
    store: EvidenceStore = InMemoryEvidenceStore()
    states = FileStateStore(root)
    trajectory = JsonlTrajectoryLogger(root / "trajectory.jsonl")
    metrics = JsonlRuntimeMetrics(root / "metrics.jsonl")
    runtime = ResearchRuntime(
        config=config,
        director=RuleBasedFrontierDirector(),
        roles={
            RoleName.RESEARCH_DIRECTOR: DemoScientist(
                threshold=threshold, seeds=seeds
            ),
            RoleName.RESEARCH_ENGINEER: DemoEngineer(
                LocalExecutor(root / "runs")
            ),
            RoleName.RESULT_ANALYST: DemoCritic(),
        },
        store=store,
        states=states,
        trajectory=trajectory,
        metrics=metrics,
        playbook=EmpiricalMLPlaybook(),
    )
    outcome = runtime.run(initial_state(), max_steps=max_steps)
    return DemoRun(
        outcome=outcome,
        store=store,
        states=states,
        trajectory_path=trajectory.path,
        metrics_path=metrics.path,
    )


# -- reporting ----------------------------------------------------------------


def _print_run(title: str, run: DemoRun) -> None:
    outcome = run.outcome
    print(f"== {title} ==")
    print(f"halted: {outcome.halt_reason}")
    print("step  action                  inv  critic  tier  outcome")
    for step, report in enumerate(outcome.reports, start=1):
        selected = report.deliberation.selected
        action = (
            selected.action.action_type.value if selected else "(declined)"
        )
        critic = "yes" if report.critic_invoked else "-"
        status = (
            next(
                (
                    a.status.value
                    for a in report.state.attempts
                    if a.id == report.record.attempt_id
                ),
                "-",
            )
            if report.record.attempt_id
            else "-"
        )
        print(
            f"{step:>4}  {action:<20}  {report.reasoning_invocations:>5}  "
            f"{critic:>6}  {report.tier.value:>4}  {status}"
        )
    state = outcome.state
    for hypothesis in state.hypotheses:
        assessment = state.current_assessment(hypothesis.id)
        verdict = assessment.verdict.value if assessment else "unassessed"
        print(f"hypothesis [{verdict}]: {hypothesis.statement}")
    print()


def experiment_report(
    run: DemoRun, action_type: ResearchActionType
) -> StepReport:
    """The first step of the run that performed ``action_type``."""
    for report in run.outcome.reports:
        selected = report.deliberation.selected
        if selected is not None and selected.action.action_type is action_type:
            return report
    raise LookupError(f"no step performed {action_type}")


def _account(label: str, report: StepReport) -> None:
    invocations = report.reasoning_invocations - (1 if report.synthesis else 0)
    critic = 1 if report.critic_invoked else 0
    suffix = " (+1 synthesis, slow loop)" if report.synthesis else ""
    print(
        f"  {label}: director 1 + executor 1 + critic {critic} "
        f"= {invocations} reasoning invocations{suffix}; "
        f"actual model calls: {report.provider_usage.calls}"
    )


def main() -> None:
    normal = run_runtime_loop(seeds=NORMAL_SEEDS, threshold=NORMAL_THRESHOLD)
    escalated = run_runtime_loop(
        seeds=ESCALATED_SEEDS, threshold=ESCALATED_THRESHOLD
    )
    _print_run("normal: ordinary result, critic never fires", normal)
    _print_run("escalated: contradictory replication, critic fires", escalated)

    print("model-call accounting, per experiment iteration")
    _account(
        "normal       ",
        experiment_report(normal, ResearchActionType.RUN_EXPERIMENT),
    )
    # The escalated run's consequential iteration is its replication step.
    _account(
        "consequential",
        experiment_report(escalated, ResearchActionType.REPLICATE),
    )
    print(f"\ntrajectories: {normal.trajectory_path} | {escalated.trajectory_path}")
    print(f"runtime metrics: {normal.metrics_path} | {escalated.metrics_path}")


if __name__ == "__main__":
    main()
