"""End-to-end walk through the architecture on a trivial question.

    ResearchState
      -> director (candidates -> utilities -> policy) selects an action
      -> ActionAttempt begun
      -> demo role produces proposals / executor produces a result
      -> CommitBundle validated and committed atomically
      -> attempt resolved inside the same transition
      -> DecisionRecord completed and logged, state snapshot persisted

The science here is a placeholder. What is being demonstrated is the shape of
the loop: every number entering the state came out of a process that ran; the
pre-registered prediction is checked mechanically when the result commits,
producing a PredictionTest per execution; the claim's standing comes from an
explicit epistemic assessment that names its method; a failed attempt leaves
work open rather than making it look done; and an attempt's entire effect
commits atomically — a bundle the transition layer rejects changes nothing.

The ``_perform`` handlers below are demo glue standing in for roles. Like real
roles, they only *read* state and return proposals — every state change goes
through the transition layer in
:mod:`autonomous_research_lab.orchestration.transitions`.

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
from autonomous_research_lab.core.commit import CommitBundle
from autonomous_research_lab.core.decision import DecisionRecord
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import ExperimentSpec, ExperimentStatus
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import (
    Comparator,
    Consistency,
    Prediction,
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
    payload_ids,
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
from autonomous_research_lab.orchestration.transitions import (
    TransitionError,
    commit_bundle,
)
from autonomous_research_lab.persistence.state_store import FileStateStore
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
    states: FileStateStore | None = None
    """Snapshot store holding every decision-boundary state of this run."""

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
    states = FileStateStore(root)
    director = ResearchDirector(
        generator=RuleBasedCandidateGenerator(),
        evaluator=HeuristicUtilityEvaluator(),
        policy=GreedySearchPolicy(),
    )
    logger = JsonlTrajectoryLogger(log_path or root / "trajectory.jsonl")
    current = state if state is not None else initial_state()
    states.persist(current)
    decisions: list[DecisionRecord] = []

    def halt(reason: str) -> LoopOutcome:
        return LoopOutcome(
            current, store, reason, tuple(decisions), logger.path, states
        )

    for _ in range(max_steps):
        decision = director.decide(current)
        action = decision.action
        if action is None:
            logger.log(decision.record)
            decisions.append(decision.record)
            return halt("policy declined every candidate")
        if action.action_type is ResearchActionType.STOP_INVESTIGATION:
            current = current.apply(action)
            states.persist(current)
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
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.SUCCEEDED,
                    produced=tuple(
                        pid for proposal in proposals for pid in payload_ids(proposal)
                    ),
                    actual_cost=cost if not cost.is_zero else estimated,
                ),
                proposals=proposals,
            )
        except Exception as exc:  # demo-grade failure handling
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.FAILED, error=str(exc), actual_cost=estimated
                ),
            )

        try:
            current = commit_bundle(current, bundle, store)
        except TransitionError as exc:
            # The whole bundle was rejected; the attempt is still unresolved
            # and the state unchanged. Resolve it as failed — on the record.
            current = commit_bundle(
                current,
                CommitBundle(
                    attempt_id=attempt.id,
                    outcome=ActionOutcome(
                        status=AttemptStatus.FAILED,
                        error=str(exc),
                        actual_cost=estimated,
                    ),
                ),
                store,
            )
        outcome = _resolved_outcome(current, attempt.id)

        current = current.apply(action)
        try:
            current = current.charge(outcome.actual_cost)
        except InsufficientBudgetError:
            states.persist(current)
            record = decision.record.completed(
                attempt_id=attempt.id, outcome=outcome, state_after_id=current.id
            )
            logger.log(record)
            decisions.append(record)
            return halt("budget exhausted mid-program")

        states.persist(current)
        record = decision.record.completed(
            attempt_id=attempt.id, outcome=outcome, state_after_id=current.id
        )
        logger.log(record)
        decisions.append(record)

    return halt(f"step limit of {max_steps} reached")


def _resolved_outcome(state: ResearchState, attempt_id: str) -> ActionOutcome:
    attempt = next(a for a in state.attempts if a.id == attempt_id)
    assert attempt.outcome is not None
    return attempt.outcome


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
            f"below is inconsistent with the prediction."
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
    test = (
        state.test_for_result(spec.prediction_id, result.id)
        if spec is not None
        else None
    )
    inconsistent = test is not None and test.consistency is Consistency.INCONSISTENT

    evidence = Evidence(
        result_id=result.id,
        spec_id=result.spec_id,
        kind=EvidenceKind.NULL_RESULT if inconsistent else EvidenceKind.MEASUREMENT,
        # Factual: what was measured, not what it means.
        observation=(
            f"Observed heads rate {heads_rate:.4f} over {n_draws} draws "
            f"(seed {result.seed})."
        ),
        metrics={"heads_rate": heads_rate, "n_draws": float(n_draws)},
    )
    return (EvidenceProposal(evidence=evidence, proposer="demo:analyst"),)


_RELATION_FOR: dict[Consistency, EvidenceRelation] = {
    Consistency.CONSISTENT: EvidenceRelation.SUPPORTS,
    Consistency.INCONSISTENT: EvidenceRelation.CONTRADICTS,
    Consistency.INCONCLUSIVE: EvidenceRelation.INCONCLUSIVE,
}


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
    test = state.test_for_result(prediction.id, evidence.result_id)
    assert test is not None

    claim = Claim(
        statement=hypothesis.statement,
        scope=spec.procedure,
        hypothesis_id=hypothesis.id,
    )
    link = EvidenceLink(
        claim_id=claim.id,
        evidence_id=evidence.id,
        relation=_RELATION_FOR[test.consistency],
        rationale=(
            f"Pre-registered prediction {prediction.id} "
            f"({prediction.metric} {prediction.comparator} "
            f"{prediction.threshold}) tested {test.consistency} against "
            f"result {test.result_id}: {test.detail}."
        ),
    )
    return (ClaimProposal(claim=claim, links=(link,), proposer="demo:analyst"),)


def _assess(state: ResearchState, action: ResearchAction) -> tuple[Proposal, ...]:
    claim = state.claim(action.targets[0])
    assert claim is not None
    hypothesis = state.hypothesis(claim.hypothesis_id) if claim.hypothesis_id else None
    tests = (
        tuple(
            test
            for prediction in state.predictions_for(hypothesis.id)
            for test in state.tests_for(prediction.id)
        )
        if hypothesis is not None
        else ()
    )
    evidence_ids = tuple(
        link.evidence_id
        for link in state.evidence_links
        if link.claim_id == claim.id
    )

    inconsistent = [t for t in tests if t.consistency is Consistency.INCONSISTENT]
    consistent = [t for t in tests if t.consistency is Consistency.CONSISTENT]
    refuted = bool(inconsistent) and not consistent
    verdict = AssessmentVerdict.REFUTED if refuted else AssessmentVerdict.UNDETERMINED
    rationale = (
        f"All {len(inconsistent)} conclusive test(s) of the pre-registered "
        f"prediction(s) were inconsistent under their stated conditions; no "
        f"auxiliary-assumption escape has been argued."
        if refuted
        else "The tests considered do not yet license a lean."
    )

    proposals: list[Proposal] = [
        AssessmentProposal(
            assessment=EpistemicAssessment(
                subject_id=claim.id,
                verdict=verdict,
                method="demo:prediction-check-v0",
                evidence_ids=evidence_ids,
                confidence=0.7 if refuted else None,
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
                    confidence=0.7 if refuted else None,
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
        assessment = state.current_assessment(hypothesis.id)
        verdict = assessment.verdict if assessment else "unassessed"
        print(f"  [{verdict}] {hypothesis.statement}")

    print("\npredictions:")
    for prediction in state.predictions:
        print(
            f"  {prediction.metric} {prediction.comparator} "
            f"{prediction.threshold} | {prediction.condition}"
        )
        for test in state.tests_for(prediction.id):
            print(f"      [{test.consistency}] {test.detail}")

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
    if outcome.states is not None:
        print(f"state snapshots: {len(outcome.states.state_ids())} persisted")


if __name__ == "__main__":
    main()
