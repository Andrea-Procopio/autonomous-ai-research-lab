"""End-to-end walk through the architecture on a trivial question.

    ResearchState
      -> ResearchDirector proposes an action
      -> ExperimentSpec
      -> LocalExecutor
      -> ExperimentResult
      -> evidence recorded
      -> ResearchState updated

The science here is a placeholder. What is being demonstrated is the shape of
the loop: that every number entering the state came out of a process that ran,
that a falsified hypothesis produces a supported claim rather than being
discarded, and that the state carries the whole trajectory afterwards.

The per-action handlers below are demo glue, not library code. In the real
system each is performed by a role backed by a model; keeping them here rather
than in the package avoids implying the package can already do this on its own.

Run with::

    python examples/minimal_loop.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.budget import (
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
)
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from autonomous_research_lab.core.hypothesis import Hypothesis, HypothesisStatus
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import EvidenceStore, InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.knowledge.graph import ClaimEvidenceGraph
from autonomous_research_lab.orchestration.rule_based import RuleBasedDirector

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPT = REPO_ROOT / "examples" / "experiments" / "coin_bias.py"

OBJECTIVE = "Determine whether the seeded draw stream used by our samplers is fair."
QUESTION = "Does the seeded Bernoulli stream deviate from a fair coin?"
BIAS_THRESHOLD = 0.55
N_DRAWS = 4000
SEED = 7


@dataclass(frozen=True, slots=True)
class LoopOutcome:
    state: ResearchState
    store: EvidenceStore
    halt_reason: str

    @property
    def actions(self) -> tuple[ResearchAction, ...]:
        return self.state.history


def initial_state(budget: ResearchBudget | None = None) -> ResearchState:
    question = ResearchQuestion(
        text=QUESTION,
        importance=(
            "Every downstream experiment draws from this stream; an undetected "
            "bias would confound all of them."
        ),
    )
    return ResearchState(
        objective=OBJECTIVE,
        questions=(question,),
        budget=budget
        if budget is not None
        else ResearchBudget(
            wall_clock_seconds=3600.0, gpu_hours=0.0, usd=10.0, model_tokens=200_000
        ),
    )


def run_minimal_loop(
    run_root: Path | str | None = None,
    *,
    max_steps: int = 12,
    state: ResearchState | None = None,
) -> LoopOutcome:
    executor = LocalExecutor(Path(run_root) if run_root else Path(mkdtemp()))
    store: EvidenceStore = InMemoryEvidenceStore()
    director = RuleBasedDirector()
    current = state if state is not None else initial_state()

    for _ in range(max_steps):
        action = director.propose(current)
        if action is None:
            return LoopOutcome(current, store, "no affordable action available")
        if action.action_type is ResearchActionType.STOP_INVESTIGATION:
            return LoopOutcome(current.apply(action), store, action.rationale)

        current, actual_cost = _perform(current, action, executor, store)
        current = current.apply(action)
        try:
            current = current.charge(actual_cost)
        except InsufficientBudgetError:
            return LoopOutcome(current, store, "budget exhausted mid-program")

    return LoopOutcome(current, store, f"step limit of {max_steps} reached")


def _perform(
    state: ResearchState,
    action: ResearchAction,
    executor: LocalExecutor,
    store: EvidenceStore,
) -> tuple[ResearchState, ResourceCost]:
    match action.action_type:
        case ResearchActionType.GENERATE_HYPOTHESIS:
            return _generate_hypothesis(state, action), action.estimated_cost
        case ResearchActionType.DESIGN_EXPERIMENT:
            return _design_experiment(state, action), action.estimated_cost
        case ResearchActionType.RUN_EXPERIMENT:
            return _run_experiment(state, action, executor, store)
        case ResearchActionType.ANALYZE:
            return _analyze(state, action, store), action.estimated_cost
        case ResearchActionType.SYNTHESIZE_FINDING:
            return _synthesize(state, action, store), action.estimated_cost
        case _:  # pragma: no cover - the rule-based director emits no others
            raise NotImplementedError(action.action_type)


def _generate_hypothesis(state: ResearchState, action: ResearchAction) -> ResearchState:
    hypothesis = Hypothesis(
        statement=(
            f"The seeded Bernoulli stream is biased toward heads, with a rate of "
            f"at least {BIAS_THRESHOLD:.2f}."
        ),
        falsification_criterion=(
            f"An observed heads rate below {BIAS_THRESHOLD:.2f} over at least "
            f"{N_DRAWS} draws falsifies the hypothesis."
        ),
        rationale="Stated in falsifiable form so the run can settle it either way.",
        assumptions=("Draws are independent.", "The generator is seeded as declared."),
        question_id=action.targets[0] if action.targets else None,
    )
    return state.upsert_hypothesis(hypothesis)


def _design_experiment(state: ResearchState, action: ResearchAction) -> ResearchState:
    hypothesis_id = action.targets[0]
    spec = ExperimentSpec(
        hypothesis_id=hypothesis_id,
        objective="Estimate the heads rate of the seeded stream.",
        procedure=(
            f"Draw {N_DRAWS} Bernoulli samples from the seeded generator and "
            f"report the observed heads rate."
        ),
        metrics=("heads_rate", "n_draws", "abs_deviation_from_half"),
        falsification_criterion=f"heads_rate < {BIAS_THRESHOLD:.2f}",
        baselines=("fair coin, rate 0.5",),
        seeds=(SEED,),
    )
    return state.add_experiment(spec)


def _run_experiment(
    state: ResearchState,
    action: ResearchAction,
    executor: LocalExecutor,
    store: EvidenceStore,
) -> tuple[ResearchState, ResourceCost]:
    spec = next(e for e in state.experiments if e.id == action.targets[0])
    job = ExperimentJob(
        spec_id=spec.id,
        command=(sys.executable, str(EXPERIMENT_SCRIPT)),
        working_dir=str(REPO_ROOT),
        config={"n_draws": N_DRAWS},
        seed=spec.seeds[0],
        timeout_seconds=120.0,
    )
    job_id = executor.submit(job)
    if not executor.status(job_id).is_terminal:  # pragma: no cover - local is sync
        raise RuntimeError(f"job {job_id} did not reach a terminal state")

    result = store.record_result(executor.collect(job_id))
    updated = state.record_result(
        ResultRef(result_id=result.id, spec_id=spec.id, status=result.status)
    )
    return updated, result.cost


def _analyze(
    state: ResearchState, action: ResearchAction, store: EvidenceStore
) -> ResearchState:
    result = store.get_result(action.targets[0])
    spec = next(e for e in state.experiments if e.id == result.spec_id)
    hypothesis = state.hypothesis(spec.hypothesis_id)
    assert hypothesis is not None

    if result.status is not ExperimentStatus.COMPLETED:
        evidence = store.record_evidence(
            Evidence(
                result_id=result.id,
                spec_id=spec.id,
                kind=EvidenceKind.FAILURE,
                observation=f"Run did not complete: {result.failure_reason}",
            )
        )
        return state.record_evidence(evidence.id).upsert_hypothesis(
            hypothesis.with_status(HypothesisStatus.INCONCLUSIVE)
        )

    heads_rate = result.metrics["heads_rate"]
    n_draws = int(result.metrics["n_draws"])
    falsified = heads_rate < BIAS_THRESHOLD

    evidence = store.record_evidence(
        Evidence(
            result_id=result.id,
            spec_id=spec.id,
            kind=EvidenceKind.NULL_RESULT if falsified else EvidenceKind.MEASUREMENT,
            # Factual: what was measured, not what it means.
            observation=(
                f"Observed heads rate {heads_rate:.4f} over {n_draws} draws "
                f"(seed {result.seed})."
            ),
            metrics={
                "heads_rate": heads_rate,
                "n_draws": float(n_draws),
            },
        )
    )
    status = HypothesisStatus.FALSIFIED if falsified else HypothesisStatus.SUPPORTED
    return state.record_evidence(evidence.id).upsert_hypothesis(
        hypothesis.with_status(status)
    )


def _synthesize(
    state: ResearchState, action: ResearchAction, store: EvidenceStore
) -> ResearchState:
    evidence = store.get_evidence(action.targets[0])
    spec = next(e for e in state.experiments if e.id == evidence.spec_id)
    hypothesis = state.hypothesis(spec.hypothesis_id)
    assert hypothesis is not None

    claim = Claim(
        statement=hypothesis.statement,
        scope=spec.procedure,
        hypothesis_id=hypothesis.id,
    )
    relation = (
        EvidenceRelation.CONTRADICTS
        if hypothesis.status is HypothesisStatus.FALSIFIED
        else EvidenceRelation.SUPPORTS
    )
    link = EvidenceLink(
        claim_id=claim.id,
        evidence_id=evidence.id,
        relation=relation,
        rationale=f"Pre-registered criterion: {spec.falsification_criterion}",
    )
    updated = state.upsert_claim(claim).link_evidence(link)
    graph = ClaimEvidenceGraph.from_state(updated, store)
    support = graph.support_for(claim.id)
    return updated.upsert_claim(claim.with_status(support.suggested_status()))


def main() -> None:
    outcome = run_minimal_loop()
    state = outcome.state
    print(f"objective: {state.objective}")
    print(f"halted: {outcome.halt_reason}\n")

    print("trajectory:")
    for step, action in enumerate(state.history, start=1):
        print(f"  {step}. {action.action_type} - {action.rationale}")

    print("\nhypotheses:")
    for hypothesis in state.hypotheses:
        print(f"  [{hypothesis.status}] {hypothesis.statement}")

    print("\nclaims:")
    graph = ClaimEvidenceGraph.from_state(state, outcome.store)
    for support in graph.all_support():
        print(f"  [{support.claim.status}] {support.claim.statement}")
        for item in support.contradicting:
            print(f"      contradicted by: {item.observation}")
        for item in support.supporting:
            print(f"      supported by:    {item.observation}")

    remaining = state.budget
    print(
        f"\nbudget left: {remaining.wall_clock_seconds:.0f}s wall-clock, "
        f"{remaining.model_tokens} model tokens"
    )


if __name__ == "__main__":
    main()
