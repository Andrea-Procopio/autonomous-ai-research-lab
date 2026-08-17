"""End-to-end walk through the architecture on a trivial question.

    ResearchState
      -> director (candidates -> utilities -> policy) selects an action
      -> ActionAttempt begun
      -> demo role produces proposals / executor produces a result
      -> transition layer validates and commits
      -> ActionAttempt resolved with an ActionOutcome
      -> DecisionRecord completed and logged

The science here is a placeholder. What is being demonstrated is the shape of
the loop: every number entering the state came out of a process that ran; the
pre-registered prediction is checked mechanically at commit time; the claim's
standing comes from an explicit epistemic assessment that names its method;
and a failed attempt leaves work open rather than making it look done.

The ``_perform`` handlers below are demo glue standing in for roles. Like real
roles, they only *read* state and return proposals — every state change goes
through :func:`~autonomous_research_lab.orchestration.transitions.commit`.

Run with::

    python examples/minimal_loop.py
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
from autonomous_research_lab.core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptStatus,
)
from autonomous_research_lab.core.budget import (
    InsufficientBudgetError,
    ResearchBudget,
    ResourceCost,
)
from autonomous_research_lab.core.claim import Claim, EvidenceLink, EvidenceRelation
from autonomous_research_lab.core.decision import DecisionRecord
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import ExperimentSpec, ExperimentStatus
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Prediction,
    PredictionStatus,
)
from autonomous_research_lab.core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    HypothesisProposal,
    PredictionProposal,
    Proposal,
    ResultProposal,
)
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.evidence.store import EvidenceStore, InMemoryEvidenceStore
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.knowledge.graph import ClaimEvidenceGraph
from autonomous_research_lab.orchestration.candidates import RuleBasedCandidateGenerator
from autonomous_research_lab.orchestration.director import ResearchDirector
from autonomous_research_lab.orchestration.evaluation import HeuristicUtilityEvaluator
from autonomous_research_lab.orchestration.trajectory import JsonlTrajectoryLogger
from autonomous_research_lab.orchestration.transitions import commit
from autonomous_research_lab.search.policy import GreedySearchPolicy

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
    decisions: tuple[DecisionRecord, ...] = ()
    log_path: Path | None = None

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
    max_steps: int = 16,
    state: ResearchState | None = None,
    log_path: Path | str | None = None,
) -> LoopOutcome:
    root = Path(run_root) if run_root else Path(mkdtemp())
    executor = LocalExecutor(root / "runs")
    store: EvidenceStore = InMemoryEvidenceStore()
    director = ResearchDirector(
        generator=RuleBasedCandidateGenerator(),
        evaluator=HeuristicUtilityEvaluator(),
        policy=GreedySearchPolicy(),
    )
    logger = JsonlTrajectoryLogger(log_path or root / "trajectory.jsonl")
    current = state if state is not None else initial_state()
    decisions: list[DecisionRecord] = []

    def halt(reason: str) -> LoopOutcome:
        return LoopOutcome(current, store, reason, tuple(decisions), logger.path)

    for _ in range(max_steps):
        decision = director.decide(current)
        action = decision.action
        if action is None:
            logger.log(decision.record)
            decisions.append(decision.record)
            return halt("policy declined every candidate")
        if action.action_type is ResearchActionType.STOP_INVESTIGATION:
            current = current.apply(action)
            record = decision.record.completed(
                attempt_id=None, outcome=None, state_after_id=current.id
            )
            logger.log(record)
            decisions.append(record)
            return halt(action.rationale)

        # Intent selected: one attempt at executing it. A retry would be a
        # new attempt; a failed one stays on the record.
        attempt = ActionAttempt(action=action).started()
        current = current.begin_attempt(attempt)
        assert decision.selected is not None
        estimated = decision.selected.utility.expected_cost

        try:
            proposals, cost = _perform(current, action, executor, store)
            for proposal in proposals:
                current = commit(current, proposal, store)
            outcome = ActionOutcome(
                status=AttemptStatus.SUCCEEDED,
                produced=_produced_ids(proposals),
                actual_cost=cost if not cost.is_zero else estimated,
            )
        except Exception as exc:  # demo-grade failure handling
            outcome = ActionOutcome(
                status=AttemptStatus.FAILED, error=str(exc), actual_cost=estimated
            )

        current = current.resolve_attempt(attempt.resolved(outcome)).apply(action)
        try:
            current = current.charge(outcome.actual_cost)
        except InsufficientBudgetError:
            record = decision.record.completed(
                attempt_id=attempt.id, outcome=outcome, state_after_id=current.id
            )
            logger.log(record)
            decisions.append(record)
            return halt("budget exhausted mid-program")

        record = decision.record.completed(
            attempt_id=attempt.id, outcome=outcome, state_after_id=current.id
        )
        logger.log(record)
        decisions.append(record)

    return halt(f"step limit of {max_steps} reached")


def _produced_ids(proposals: tuple[Proposal, ...]) -> tuple[str, ...]:
    produced: list[str] = []
    for proposal in proposals:
        match proposal:
            case HypothesisProposal():
                produced.append(proposal.hypothesis.id)
            case PredictionProposal():
                produced.append(proposal.prediction.id)
            case ExperimentProposal():
                produced.append(proposal.spec.id)
            case ResultProposal():
                produced.append(proposal.result.id)
            case EvidenceProposal():
                produced.append(proposal.evidence.id)
            case ClaimProposal():
                produced.append(proposal.claim.id)
            case AssessmentProposal():
                produced.append(proposal.assessment.id)
    return tuple(produced)


# -- demo roles ---------------------------------------------------------------
# Each handler stands in for a role: it reads state and returns proposals plus
# the cost actually incurred (zero means "charge the estimate instead").


def _perform(
    state: ResearchState,
    action: ResearchAction,
    executor: LocalExecutor,
    store: EvidenceStore,
) -> tuple[tuple[Proposal, ...], ResourceCost]:
    match action.action_type:
        case ResearchActionType.GENERATE_HYPOTHESIS:
            return _generate_hypothesis(action), ResourceCost()
        case ResearchActionType.DERIVE_PREDICTION:
            return _derive_prediction(action), ResourceCost()
        case ResearchActionType.DESIGN_EXPERIMENT:
            return _design_experiment(state, action), ResourceCost()
        case ResearchActionType.RUN_EXPERIMENT:
            return _run_experiment(state, action, executor)
        case ResearchActionType.ANALYZE:
            return _analyze(state, action, store), ResourceCost()
        case ResearchActionType.SYNTHESIZE_FINDING:
            return _synthesize(state, action, store), ResourceCost()
        case ResearchActionType.ASSESS_CLAIM:
            return _assess(state, action), ResourceCost()
        case _:  # pragma: no cover - the rule-based generator emits no others
            raise NotImplementedError(action.action_type)


def _generate_hypothesis(action: ResearchAction) -> tuple[Proposal, ...]:
    hypothesis = Hypothesis(
        statement=(
            f"The seeded Bernoulli stream is biased toward heads, with a rate of "
            f"at least {BIAS_THRESHOLD:.2f}."
        ),
        rationale=(
            "Deliberately false, so the loop demonstrates falsification rather "
            "than confirmation."
        ),
        assumptions=("Draws are independent.", "The generator is seeded as declared."),
        question_id=action.targets[0] if action.targets else None,
    )
    return (HypothesisProposal(hypothesis=hypothesis, proposer="demo:generator"),)


def _derive_prediction(action: ResearchAction) -> tuple[Proposal, ...]:
    prediction = Prediction(
        hypothesis_id=action.targets[0],
        condition=f"{N_DRAWS} draws from the seeded generator, seed {SEED}",
        metric="heads_rate",
        comparator=Comparator.GREATER_OR_EQUAL,
        threshold=BIAS_THRESHOLD,
        expectation=(
            f"The observed heads rate is at least {BIAS_THRESHOLD:.2f}; anything "
            f"below fails the prediction."
        ),
    )
    return (PredictionProposal(prediction=prediction, proposer="demo:generator"),)


def _design_experiment(
    state: ResearchState, action: ResearchAction
) -> tuple[Proposal, ...]:
    prediction = state.prediction(action.targets[0])
    assert prediction is not None
    spec = ExperimentSpec(
        prediction_id=prediction.id,
        objective="Estimate the heads rate of the seeded stream.",
        procedure=(
            f"Draw {N_DRAWS} Bernoulli samples from the seeded generator and "
            f"report the observed heads rate."
        ),
        metrics=("heads_rate", "n_draws", "abs_deviation_from_half"),
        baselines=("fair coin, rate 0.5",),
        seeds=(SEED,),
    )
    return (ExperimentProposal(spec=spec, proposer="demo:designer"),)


def _run_experiment(
    state: ResearchState, action: ResearchAction, executor: LocalExecutor
) -> tuple[tuple[Proposal, ...], ResourceCost]:
    spec = state.experiment(action.targets[0])
    assert spec is not None
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
    result = executor.collect(job_id)
    return (ResultProposal(result=result, proposer="executor:local"),), result.cost


def _analyze(
    state: ResearchState, action: ResearchAction, store: EvidenceStore
) -> tuple[Proposal, ...]:
    result = store.get_result(action.targets[0])

    if result.status is not ExperimentStatus.COMPLETED:
        evidence = Evidence(
            result_id=result.id,
            spec_id=result.spec_id,
            kind=EvidenceKind.FAILURE,
            observation=f"Run did not complete: {result.failure_reason}",
        )
        return (EvidenceProposal(evidence=evidence, proposer="demo:analyst"),)

    heads_rate = result.metrics["heads_rate"]
    n_draws = int(result.metrics["n_draws"])
    spec = state.experiment(result.spec_id)
    prediction = state.prediction(spec.prediction_id) if spec else None
    fails_prediction = prediction is not None and not prediction.check(heads_rate)

    evidence = Evidence(
        result_id=result.id,
        spec_id=result.spec_id,
        kind=EvidenceKind.NULL_RESULT if fails_prediction else EvidenceKind.MEASUREMENT,
        # Factual: what was measured, not what it means.
        observation=(
            f"Observed heads rate {heads_rate:.4f} over {n_draws} draws "
            f"(seed {result.seed})."
        ),
        metrics={"heads_rate": heads_rate, "n_draws": float(n_draws)},
    )
    return (EvidenceProposal(evidence=evidence, proposer="demo:analyst"),)


def _synthesize(
    state: ResearchState, action: ResearchAction, store: EvidenceStore
) -> tuple[Proposal, ...]:
    evidence = store.get_evidence(action.targets[0])
    spec = state.experiment(evidence.spec_id)
    assert spec is not None
    prediction = state.prediction(spec.prediction_id)
    assert prediction is not None
    hypothesis = state.hypothesis(prediction.hypothesis_id)
    assert hypothesis is not None

    claim = Claim(
        statement=hypothesis.statement,
        scope=spec.procedure,
        hypothesis_id=hypothesis.id,
    )
    relation = (
        EvidenceRelation.CONTRADICTS
        if prediction.status is PredictionStatus.FAILED
        else EvidenceRelation.SUPPORTS
    )
    link = EvidenceLink(
        claim_id=claim.id,
        evidence_id=evidence.id,
        relation=relation,
        rationale=(
            f"Pre-registered prediction {prediction.id} "
            f"({prediction.metric} {prediction.comparator} "
            f"{prediction.threshold}) resolved {prediction.status}."
        ),
    )
    return (ClaimProposal(claim=claim, links=(link,), proposer="demo:analyst"),)


def _assess(state: ResearchState, action: ResearchAction) -> tuple[Proposal, ...]:
    claim = state.claim(action.targets[0])
    assert claim is not None
    hypothesis = state.hypothesis(claim.hypothesis_id) if claim.hypothesis_id else None
    predictions = (
        state.predictions_for(hypothesis.id) if hypothesis is not None else ()
    )
    prediction = predictions[0] if predictions else None
    evidence_ids = tuple(
        link.evidence_id
        for link in state.evidence_links
        if link.claim_id == claim.id
    )

    failed = prediction is not None and prediction.status is PredictionStatus.FAILED
    verdict = AssessmentVerdict.REFUTED if failed else AssessmentVerdict.UNDETERMINED
    rationale = (
        "The single pre-registered prediction failed under its stated "
        "condition; no auxiliary-assumption escape has been argued."
        if failed
        else "The evidence considered does not yet license a lean."
    )

    proposals: list[Proposal] = [
        AssessmentProposal(
            assessment=EpistemicAssessment(
                subject_id=claim.id,
                verdict=verdict,
                method="demo:prediction-check-v0",
                evidence_ids=evidence_ids,
                confidence=0.7 if failed else None,
                scope=claim.scope,
                rationale=rationale,
            ),
            proposer="demo:assessor",
        )
    ]
    if hypothesis is not None:
        proposals.append(
            AssessmentProposal(
                assessment=EpistemicAssessment(
                    subject_id=hypothesis.id,
                    verdict=verdict,
                    method="demo:prediction-check-v0",
                    evidence_ids=evidence_ids,
                    confidence=0.7 if failed else None,
                    rationale=rationale,
                ),
                proposer="demo:assessor",
            )
        )
    return tuple(proposals)


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

    print("\npredictions:")
    for prediction in state.predictions:
        print(
            f"  [{prediction.status}] {prediction.metric} "
            f"{prediction.comparator} {prediction.threshold} | {prediction.condition}"
        )

    print("\nclaims:")
    graph = ClaimEvidenceGraph.from_state(state, outcome.store)
    for entry in graph.all_claims():
        assessment = state.current_assessment(entry.claim.id)
        verdict = assessment.verdict if assessment else "unassessed"
        method = f" (method: {assessment.method})" if assessment else ""
        print(f"  [{verdict}]{method} {entry.claim.statement}")
        for item in entry.contradicting:
            print(f"      contradicted by: {item.observation}")
        for item in entry.supporting:
            print(f"      supported by:    {item.observation}")

    remaining = state.budget
    print(
        f"\nbudget left: {remaining.wall_clock_seconds:.0f}s wall-clock, "
        f"{remaining.model_tokens} model tokens"
    )
    print(f"trajectory log: {outcome.log_path}")


if __name__ == "__main__":
    main()
