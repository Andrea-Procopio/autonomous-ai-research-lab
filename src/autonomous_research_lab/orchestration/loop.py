"""The research runtime: the loop that keeps model calls sparse.

One step of the fast loop::

    ResearchState
      -> ResearchFrontier (derived view)
      -> director deliberates once: candidates + valuations + selection
      -> deterministic role routing
      -> one role invocation performs the action, proposals come back
      -> CommitBundle commits atomically (mechanical prediction test included)
      -> deterministic validation + deterministic evidence reading (Tier 0)
      -> critic trigger evaluated (deterministic); critic invoked only if it fires
      -> synthesis trigger evaluated; slow loop runs only when due
      -> state persisted, decision + runtime metrics logged

The invariant this module exists to hold: **an ordinary experiment iteration
costs one director invocation and one executor invocation — nothing else.**
Validation, evidence transcription, prediction checking, routing, and
trigger evaluation are all code. A critic costs a third invocation only when
a deterministic trigger says the result is consequential, and the slow
synthesis loop is the same director in a stronger reasoning mode, on a
deterministic cadence.

Roles are injected per seat (scientist / executor / critic), so the same
loop runs mock roles today and model-backed roles later without changing
shape. Everything a role sees arrives through its ``RoleInvocation`` — a
projection built here, never the raw state — which is also what keeps
executors short-lived and narrow while the director stays long-lived.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core.actions import ResearchAction, ResearchActionType
from ..core.attempt import ActionAttempt, ActionOutcome, AttemptStatus
from ..core.budget import NO_COST, InsufficientBudgetError, ResourceCost
from ..core.commit import CommitBundle
from ..core.decision import DecisionRecord
from ..core.experiment import ExperimentResult
from ..core.proposals import EvidenceProposal, ResultProposal, payload_ids
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore
from ..persistence.state_store import FileStateStore
from ..roles.base import ResearchRole, RoleContext, RoleInvocation, RoleName
from ..runtime.config import RuntimeConfig
from ..runtime.escalation import (
    EscalationPolicy,
    EscalationSignals,
    Level,
    ReasoningTier,
)
from ..runtime.frontier import ResearchFrontier, build_frontier, find_contradictions
from ..runtime.metrics import MetricsSink, StepMetrics
from ..runtime.playbook import Playbook, PlaybookAdvice
from ..runtime.validation import (
    ValidationReport,
    evidence_from_result,
    validate_result,
)
from .critic_trigger import CriticTrigger
from .director import Deliberation, FrontierDirector, deliberation_record
from .routing import expected_proposals, route
from .synthesis import SynthesisReview, SynthesisTrigger
from .trajectory import JsonlTrajectoryLogger
from .transitions import TransitionError, commit, commit_bundle

_READER = "runtime:deterministic-reader:v1"


@dataclass(frozen=True, slots=True)
class StepReport:
    """Everything one step decided, did, checked, and spent."""

    record: DecisionRecord
    state: ResearchState
    deliberation: Deliberation
    tier: ReasoningTier
    llm_calls: int
    validation: ValidationReport | None = None
    critic_reasons: tuple[str, ...] = ()
    critic_invoked: bool = False
    synthesis: SynthesisReview | None = None
    halt_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunOutcome:
    state: ResearchState
    halt_reason: str
    reports: tuple[StepReport, ...]


class MissingRoleError(LookupError):
    """Raised when routing selects a seat no role was registered for."""


@dataclass
class ResearchRuntime:
    """The wiring of one research program's runtime. Long-lived, like the
    director it hosts; the roles it invokes are given one narrow assignment
    at a time and nothing else."""

    config: RuntimeConfig
    director: FrontierDirector
    roles: Mapping[RoleName, ResearchRole]
    store: EvidenceStore
    states: FileStateStore | None = None
    trajectory: JsonlTrajectoryLogger | None = None
    metrics: MetricsSink | None = None
    playbook: Playbook | None = None
    critic_trigger: CriticTrigger = field(default_factory=CriticTrigger)
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    synthesis_trigger: SynthesisTrigger | None = None
    _results_since_synthesis: int = field(default=0, init=False)
    _notes: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.synthesis_trigger is None:
            self.synthesis_trigger = SynthesisTrigger(
                every=self.config.synthesis_every
            )

    def run(self, state: ResearchState, *, max_steps: int = 32) -> RunOutcome:
        self._persist(state)
        reports: list[StepReport] = []
        for _ in range(max_steps):
            report = self.step(state)
            reports.append(report)
            state = report.state
            if report.halt_reason is not None:
                return RunOutcome(state, report.halt_reason, tuple(reports))
        return RunOutcome(
            state, f"step limit of {max_steps} reached", tuple(reports)
        )

    def step(self, state: ResearchState) -> StepReport:
        """One fast-loop iteration, with the slow loop run when due."""
        started = time.monotonic()
        frontier = build_frontier(
            state,
            recent_results=self.config.recent_results,
            open_decisions=self._notes,
        )
        self._notes = ()

        advice: PlaybookAdvice | None = None
        if self.config.playbook_enabled and self.playbook is not None:
            advice = self.playbook.advise(frontier)

        tier = max(
            self.config.director_tier_floor,
            self.escalation.tier_for(_signals(frontier)),
        )
        deliberation = self.director.deliberate(
            frontier,
            advice=advice,
            tier=tier,
            max_candidates=self.config.max_candidates,
        )
        llm_calls = 1  # the deliberation is the one mandatory reasoning call
        record = deliberation_record(
            deliberation, state_id=state.id, director=self.director.name
        )
        selected = deliberation.selected

        if selected is None:
            return self._halted(
                state, record, deliberation, tier, llm_calls,
                started, "director declined every candidate",
            )
        action = selected.action
        if action.action_type is ResearchActionType.STOP_INVESTIGATION:
            return self._stop(
                state, record, deliberation, tier, llm_calls, started, action
            )

        seat = route(action.action_type)
        performer = self.roles.get(seat)
        if performer is None:
            raise MissingRoleError(
                f"action {action.action_type} routes to {seat}, but no role "
                f"is registered for that seat"
            )

        attempt = ActionAttempt(action=action).started()
        state = state.begin_attempt(attempt)
        estimated = selected.valuation.expected_cost
        invocation = RoleInvocation(
            role=seat,
            assignment=action,
            context=_context_for(state, self.store, action),
            allowed_actions=frozenset({action.action_type}),
            expected_output=expected_proposals(action.action_type),
            budget=estimated,
        )

        failures = 0
        try:
            proposals = performer.perform(invocation)
            llm_calls += 1
            rejected = [p for p in proposals if not invocation.permits(p)]
            if rejected:
                raise TransitionError(
                    f"role {seat} returned proposal kind(s) outside its "
                    f"output contract: "
                    f"{', '.join(type(p).__name__ for p in rejected)}"
                )
            results = tuple(
                p.result for p in proposals if isinstance(p, ResultProposal)
            )
            actual = _actual_cost(results, estimated)
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.SUCCEEDED,
                    produced=tuple(
                        pid for p in proposals for pid in payload_ids(p)
                    ),
                    actual_cost=actual,
                ),
                proposals=proposals,
            )
        except TransitionError as exc:
            failures = 1
            bundle = _failed_bundle(attempt.id, str(exc), estimated)
            results = ()
        except Exception as exc:  # a role failing is an outcome, not a crash
            failures = 1
            llm_calls += 1  # the invocation happened even though it failed
            bundle = _failed_bundle(attempt.id, str(exc), estimated)
            results = ()

        contradictions_before = len(find_contradictions(state))
        try:
            state = commit_bundle(state, bundle, self.store)
        except TransitionError as exc:
            failures += 1
            results = ()
            state = commit_bundle(
                state, _failed_bundle(attempt.id, str(exc), estimated), self.store
            )
        outcome = _outcome_of(state, attempt.id)

        # -- Tier 0 aftermath of an executed result --------------------------
        validation: ValidationReport | None = None
        critic_reasons: tuple[str, ...] = ()
        critic_invoked = False
        if outcome.status is AttemptStatus.SUCCEEDED and results:
            result = results[0]
            self._results_since_synthesis += len(results)
            state, validation = self._read_result(state, result)
            critic_reasons = self._critic_reasons(state, result, validation)
            if critic_reasons and self.config.critic_enabled:
                state, invoked_calls = self._invoke_critic(
                    state, result, critic_reasons, validation
                )
                llm_calls += invoked_calls
                critic_invoked = invoked_calls > 0

        state = state.apply(action)
        try:
            state = state.charge(outcome.actual_cost)
        except InsufficientBudgetError:
            return self._finish(
                state, record, deliberation, tier, llm_calls, started,
                attempt_id=attempt.id, outcome=outcome, seat=seat,
                validation=validation, critic_reasons=critic_reasons,
                critic_invoked=critic_invoked, failures=failures,
                results=results, halt_reason="budget exhausted mid-program",
            )

        synthesis = self._maybe_synthesize(
            state,
            new_contradiction=len(find_contradictions(state))
            > contradictions_before,
            stopping=False,
        )
        if synthesis is not None:
            llm_calls += 1

        return self._finish(
            state, record, deliberation, tier, llm_calls, started,
            attempt_id=attempt.id, outcome=outcome, seat=seat,
            validation=validation, critic_reasons=critic_reasons,
            critic_invoked=critic_invoked, failures=failures,
            results=results, synthesis=synthesis, halt_reason=None,
        )

    # -- deterministic aftermath --------------------------------------------

    def _read_result(
        self, state: ResearchState, result: ExperimentResult
    ) -> tuple[ResearchState, ValidationReport]:
        """Validate and transcribe one committed result. Zero model calls:
        the reading reuses the mechanical prediction test the commit already
        produced, and the evidence proposal is attributed to the runtime."""
        spec = state.experiment(result.spec_id)
        assert spec is not None  # the bundle could not have committed otherwise
        prediction = state.prediction(spec.prediction_id)
        validation = validate_result(spec, result, prediction=prediction)
        test = (
            state.test_for_result(prediction.id, result.id)
            if prediction is not None
            else None
        )
        evidence = evidence_from_result(result, test=test)
        state = commit(
            state, EvidenceProposal(evidence=evidence, proposer=_READER), self.store
        )
        return state, validation

    def _critic_reasons(
        self,
        state: ResearchState,
        result: ExperimentResult,
        validation: ValidationReport,
    ) -> tuple[str, ...]:
        spec = state.experiment(result.spec_id)
        prediction = (
            state.prediction(spec.prediction_id) if spec is not None else None
        )
        test = (
            state.test_for_result(prediction.id, result.id)
            if prediction is not None
            else None
        )
        return self.critic_trigger.reasons(
            state, result=result, validation=validation, test=test
        )

    def _invoke_critic(
        self,
        state: ResearchState,
        result: ExperimentResult,
        reasons: tuple[str, ...],
        validation: ValidationReport,
    ) -> tuple[ResearchState, int]:
        critic = self.roles.get(RoleName.RESULT_ANALYST)
        if critic is None:
            return state, 0
        action = ResearchAction(
            action_type=ResearchActionType.ANALYZE,
            rationale="; ".join(reasons),
            targets=(result.id,),
        )
        attempt = ActionAttempt(action=action).started()
        state = state.begin_attempt(attempt)
        invocation = RoleInvocation(
            role=RoleName.RESULT_ANALYST,
            assignment=action,
            context=_critic_context(state, self.store, result, reasons, validation),
            allowed_actions=frozenset({ResearchActionType.ANALYZE}),
            expected_output=expected_proposals(ResearchActionType.ANALYZE),
        )
        try:
            proposals = critic.perform(invocation)
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.SUCCEEDED,
                    produced=tuple(
                        pid for p in proposals for pid in payload_ids(p)
                    ),
                ),
                proposals=proposals,
            )
            state = commit_bundle(state, bundle, self.store)
        except Exception as exc:
            # A critic that fails leaves the record showing it was asked.
            state = commit_bundle(
                state, _failed_bundle(attempt.id, str(exc), NO_COST), self.store
            )
        return state, 1

    def _maybe_synthesize(
        self,
        state: ResearchState,
        *,
        new_contradiction: bool,
        stopping: bool,
    ) -> SynthesisReview | None:
        if not self.config.synthesis_enabled:
            return None
        assert self.synthesis_trigger is not None
        reasons = self.synthesis_trigger.reasons(
            results_since_synthesis=self._results_since_synthesis,
            new_contradiction=new_contradiction,
            stopping=stopping,
        )
        if not reasons:
            return None
        review = self.director.synthesize(
            build_frontier(state, recent_results=self.config.recent_results),
            tier=max(ReasoningTier.STRONG, self.config.director_tier_floor),
        )
        self._results_since_synthesis = 0
        self._notes = (
            f"last synthesis ({'; '.join(reasons)}): {review.summary}",
        )
        return review

    # -- bookkeeping ---------------------------------------------------------

    def _stop(
        self,
        state: ResearchState,
        record: DecisionRecord,
        deliberation: Deliberation,
        tier: ReasoningTier,
        llm_calls: int,
        started: float,
        action: ResearchAction,
    ) -> StepReport:
        synthesis = self._maybe_synthesize(
            state, new_contradiction=False, stopping=True
        )
        if synthesis is not None:
            llm_calls += 1
        state = state.apply(action)
        return self._finish(
            state, record, deliberation, tier, llm_calls, started,
            attempt_id=None, outcome=None, seat=None,
            validation=None, critic_reasons=(), critic_invoked=False,
            failures=0, results=(), synthesis=synthesis,
            halt_reason=action.rationale,
        )

    def _halted(
        self,
        state: ResearchState,
        record: DecisionRecord,
        deliberation: Deliberation,
        tier: ReasoningTier,
        llm_calls: int,
        started: float,
        reason: str,
    ) -> StepReport:
        return self._finish(
            state, record, deliberation, tier, llm_calls, started,
            attempt_id=None, outcome=None, seat=None,
            validation=None, critic_reasons=(), critic_invoked=False,
            failures=0, results=(), halt_reason=reason,
        )

    def _finish(
        self,
        state: ResearchState,
        record: DecisionRecord,
        deliberation: Deliberation,
        tier: ReasoningTier,
        llm_calls: int,
        started: float,
        *,
        attempt_id: str | None,
        outcome: ActionOutcome | None,
        seat: RoleName | None,
        validation: ValidationReport | None,
        critic_reasons: tuple[str, ...],
        critic_invoked: bool,
        failures: int,
        results: tuple[ExperimentResult, ...],
        synthesis: SynthesisReview | None = None,
        halt_reason: str | None,
    ) -> StepReport:
        self._persist(state)
        completed = record.completed(
            attempt_id=attempt_id,
            outcome=outcome,
            state_after_id=state.id,
            assigned_role=seat.value if seat is not None else None,
        )
        if self.trajectory is not None:
            self.trajectory.log(completed)
        if self.metrics is not None:
            selected = deliberation.selected
            action_type = (
                selected.action.action_type.value if selected else "none"
            )
            self.metrics.log(
                StepMetrics(
                    decision_id=completed.id,
                    action_type=action_type,
                    outcome_status=outcome.status.value if outcome else "none",
                    reasoning_tier=tier,
                    llm_calls=llm_calls,
                    wall_clock_seconds=time.monotonic() - started,
                    experiment_seconds=sum(r.runtime_seconds for r in results),
                    estimated_usd=(
                        selected.valuation.expected_cost.usd if selected else 0.0
                    ),
                    failures=failures,
                    critic_invoked=critic_invoked,
                    critic_reasons=critic_reasons,
                    synthesis_invoked=synthesis is not None,
                    rationale=deliberation.reasoning,
                    notes=tuple(
                        f"validation:{check.name}"
                        for check in (validation.failures if validation else ())
                    ),
                )
            )
        return StepReport(
            record=completed,
            state=state,
            deliberation=deliberation,
            tier=tier,
            llm_calls=llm_calls,
            validation=validation,
            critic_reasons=critic_reasons,
            critic_invoked=critic_invoked,
            synthesis=synthesis,
            halt_reason=halt_reason,
        )

    def _persist(self, state: ResearchState) -> None:
        if self.states is not None:
            self.states.persist(state)


# -- helpers -----------------------------------------------------------------


def _signals(frontier: ResearchFrontier) -> EscalationSignals:
    """Deterministic escalation signals read off the frontier."""
    return EscalationSignals(
        importance=Level.HIGH if frontier.contradictions else Level.MEDIUM,
        uncertainty=(
            Level.HIGH if len(frontier.active_hypotheses) > 1 else Level.MEDIUM
        ),
        downstream_cost=_pending_cost(frontier),
        conflicting_evidence=bool(frontier.contradictions),
    )


def _pending_cost(frontier: ResearchFrontier) -> ResourceCost:
    total = ResourceCost()
    for spec in frontier.pending_experiments:
        total = total + spec.estimated_cost
    return total


def _actual_cost(
    results: tuple[ExperimentResult, ...], estimated: ResourceCost
) -> ResourceCost:
    if not results:
        return estimated
    total = ResourceCost()
    for result in results:
        total = total + result.cost
    return total if not total.is_zero else estimated


def _failed_bundle(
    attempt_id: str, error: str, cost: ResourceCost
) -> CommitBundle:
    return CommitBundle(
        attempt_id=attempt_id,
        outcome=ActionOutcome(
            status=AttemptStatus.FAILED, error=error, actual_cost=cost
        ),
    )


def _outcome_of(state: ResearchState, attempt_id: str) -> ActionOutcome:
    attempt = next(a for a in state.attempts if a.id == attempt_id)
    assert attempt.outcome is not None
    return attempt.outcome


def _context_for(
    state: ResearchState, store: EvidenceStore, action: ResearchAction
) -> RoleContext:
    """The projection each seat receives: exactly what the assignment needs.

    This is where worker lifetime is separated from lab lifetime — an
    executor sees a spec and its prior runs, never the research history.
    """
    target = action.targets[0] if action.targets else None
    match action.action_type:
        case ResearchActionType.GENERATE_HYPOTHESIS:
            return RoleContext(
                objective=state.objective,
                questions=state.questions,
                hypotheses=state.hypotheses,
            )
        case ResearchActionType.DERIVE_PREDICTION:
            hypothesis = state.hypothesis(target) if target else None
            return RoleContext(
                objective=state.objective,
                hypotheses=(hypothesis,) if hypothesis else (),
            )
        case ResearchActionType.DESIGN_EXPERIMENT:
            prediction = state.prediction(target) if target else None
            owner = (
                state.hypothesis(prediction.hypothesis_id) if prediction else None
            )
            return RoleContext(
                objective=state.objective,
                hypotheses=(owner,) if owner else (),
                predictions=(prediction,) if prediction else (),
            )
        case ResearchActionType.RUN_EXPERIMENT | ResearchActionType.REPLICATE:
            spec = state.experiment(target) if target else None
            prior = (
                tuple(
                    store.get_result(ref.result_id)
                    for ref in state.results_for(spec.id)
                )
                if spec is not None
                else ()
            )
            return RoleContext(
                objective=state.objective,
                experiments=(spec,) if spec else (),
                results=prior,
            )
        case ResearchActionType.SYNTHESIZE_FINDING:
            return _synthesis_context(state, store, target)
        case ResearchActionType.ASSESS_CLAIM:
            return _assessment_context(state, store, target)
        case _:
            return RoleContext(objective=state.objective)


def _synthesis_context(
    state: ResearchState, store: EvidenceStore, evidence_id: str | None
) -> RoleContext:
    if evidence_id is None:
        return RoleContext(objective=state.objective)
    evidence = store.get_evidence(evidence_id)
    spec = state.experiment(evidence.spec_id)
    prediction = state.prediction(spec.prediction_id) if spec else None
    hypothesis = (
        state.hypothesis(prediction.hypothesis_id) if prediction else None
    )
    tests = state.tests_for(prediction.id) if prediction else ()
    return RoleContext(
        objective=state.objective,
        hypotheses=(hypothesis,) if hypothesis else (),
        predictions=(prediction,) if prediction else (),
        prediction_tests=tuple(tests),
        experiments=(spec,) if spec else (),
        evidence=(evidence,),
    )


def _assessment_context(
    state: ResearchState, store: EvidenceStore, claim_id: str | None
) -> RoleContext:
    claim = state.claim(claim_id) if claim_id else None
    if claim is None:
        return RoleContext(objective=state.objective)
    links = tuple(
        link for link in state.evidence_links if link.claim_id == claim.id
    )
    hypothesis = (
        state.hypothesis(claim.hypothesis_id) if claim.hypothesis_id else None
    )
    predictions = (
        state.predictions_for(hypothesis.id) if hypothesis is not None else ()
    )
    tests = tuple(
        test
        for prediction in predictions
        for test in state.tests_for(prediction.id)
    )
    return RoleContext(
        objective=state.objective,
        hypotheses=(hypothesis,) if hypothesis else (),
        predictions=tuple(predictions),
        prediction_tests=tests,
        evidence=tuple(store.get_evidence(link.evidence_id) for link in links),
        claims=(claim,),
        evidence_links=links,
        assessments=state.assessments,
    )


def _critic_context(
    state: ResearchState,
    store: EvidenceStore,
    result: ExperimentResult,
    reasons: tuple[str, ...],
    validation: ValidationReport,
) -> RoleContext:
    spec = state.experiment(result.spec_id)
    prediction = state.prediction(spec.prediction_id) if spec else None
    hypothesis = (
        state.hypothesis(prediction.hypothesis_id) if prediction else None
    )
    tests = state.tests_for(prediction.id) if prediction else ()
    family = (
        tuple(
            store.get_result(ref.result_id)
            for ref in state.results_for(spec.id)
        )
        if spec is not None
        else (result,)
    )
    family_ids = {r.id for r in family}
    evidence = tuple(
        e
        for e in (store.get_evidence(eid) for eid in state.evidence_ids)
        if e.result_id in family_ids
    )
    return RoleContext(
        objective=state.objective,
        hypotheses=(hypothesis,) if hypothesis else (),
        predictions=(prediction,) if prediction else (),
        prediction_tests=tuple(tests),
        experiments=(spec,) if spec else (),
        results=family,
        evidence=evidence,
        assessments=state.assessments,
        notes=(
            *reasons,
            *(
                f"validation failure: {check.name} ({check.detail})"
                for check in validation.failures
            ),
        ),
    )
