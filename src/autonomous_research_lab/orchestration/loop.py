"""The research runtime: the loop that keeps reasoning invocations sparse.

One step of the fast loop::

    ResearchState
      -> ResearchFrontier (derived view)
      -> budget check: an unaffordable action is never started
      -> director deliberates once: candidates + valuations + selection
      -> deterministic role routing
      -> one role invocation performs the action, proposals come back
      -> Tier-0 validation gate: every returned result is checked BEFORE
         anything commits — cardinality, assignment correspondence, declared
         metrics, finite values, seed, artifact integrity
      -> CommitBundle commits atomically (mechanical prediction test included)
      -> deterministic evidence reading of valid completed results (Tier 0)
      -> critic trigger evaluated (deterministic, scientific reasons only)
      -> synthesis trigger evaluated; slow loop runs only when due
      -> cost reconciled against the budget — work is never committed unbilled
      -> state persisted, decision + runtime metrics logged

The enforceable invariant: **an ordinary experiment step makes exactly two
reasoning-seat invocations — one ``director.deliberate()`` and one
``performer.perform()``.** Validation, evidence transcription, prediction
checking, routing, and trigger evaluation are code. A critic adds a third
invocation only when a deterministic trigger finds a *scientific* reason,
and synthesis is the same director in a stronger mode on a deterministic
cadence. What the loop cannot enforce — and does not claim to — is how many
provider calls a model-backed role makes inside one invocation; those are
recorded separately, from provider reports (:class:`~autonomous_research_lab.
runtime.metrics.UsageSource`), and are zero for rule-based roles.

Failure taxonomy, kept deliberately separate:

* a result failing the deterministic gate is an **engineering failure**: the
  attempt fails, nothing enters scientific state, the run directory is
  preserved, and the director sees a deterministic note next step — never a
  critic, because no model opinion can override arithmetic;
* a failed/cancelled *execution* is an honest execution record: it commits
  with inconclusive scientific standing, and repeated failures of one
  experiment raise a deterministic engineering note;
* a *scientifically valid* consequential result — contradiction, challenged
  standing, large effect — is what earns a critic.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core.actions import ResearchAction, ResearchActionType
from ..core.attempt import ActionAttempt, ActionOutcome, AttemptStatus
from ..core.budget import NO_COST, ResearchBudget, ResourceCost
from ..core.commit import CommitBundle
from ..core.decision import DecisionRecord
from ..core.experiment import ExperimentResult, ExperimentSpec
from ..core.proposals import (
    EvidenceProposal,
    ExperimentProposal,
    Proposal,
    ResultProposal,
    payload_ids,
)
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
from ..runtime.metrics import (
    NO_USAGE,
    MetricsSink,
    ProviderUsage,
    StepMetrics,
    UsageSource,
)
from ..runtime.playbook import Playbook, PlaybookAdvice
from ..runtime.validation import (
    ValidationCheck,
    ValidationReport,
    evidence_from_result,
    validate_result,
    verify_artifact_integrity,
)
from .critic_trigger import CriticTrigger
from .director import Deliberation, FrontierDirector, deliberation_record
from .routing import expected_proposals, route
from .synthesis import SynthesisReview, SynthesisTrigger
from .trajectory import JsonlTrajectoryLogger
from .transitions import TransitionError, commit, commit_bundle

_READER = "runtime:deterministic-reader:v1"

#: Actions whose invocation must return exactly one result. An executor
#: assigned one run that reports zero or several is out of contract.
_SINGLE_RESULT_ACTIONS = frozenset(
    {
        ResearchActionType.RUN_EXPERIMENT,
        ResearchActionType.REPLICATE,
        ResearchActionType.TEST_BASELINE,
        ResearchActionType.SCALE_EXPERIMENT,
    }
)


class ValidationGateError(Exception):
    """A returned result failed the deterministic pre-commit gate."""

    def __init__(self, message: str, reports: tuple[ValidationReport, ...]):
        super().__init__(message)
        self.reports = reports


@dataclass(frozen=True, slots=True)
class StepReport:
    """Everything one step decided, did, checked, and spent."""

    record: DecisionRecord
    state: ResearchState
    deliberation: Deliberation
    tier: ReasoningTier
    reasoning_invocations: int
    provider_usage: ProviderUsage = NO_USAGE
    validation: tuple[ValidationReport, ...] = ()
    critic_reasons: tuple[str, ...] = ()
    critic_invoked: bool = False
    synthesis: SynthesisReview | None = None
    notes: tuple[str, ...] = ()
    """Deterministic runtime notes raised this step (engineering failures,
    repeated execution failures, budget overruns)."""

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
    usage: UsageSource | None = None
    """Provider-usage reporter, when a provider adapter exists. Drained once
    per step into the metrics record; ``None`` means actual model usage is
    honestly recorded as zero."""

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
        step_notes: list[str] = []
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
        invocations = 1  # the deliberation is the one mandatory invocation
        record = deliberation_record(
            deliberation, state_id=state.id, director=self.director.name
        )
        selected = deliberation.selected

        if selected is None:
            return self._halted(
                state, record, deliberation, tier, invocations,
                started, "director declined every candidate",
            )
        action = selected.action
        if action.action_type is ResearchActionType.STOP_INVESTIGATION:
            return self._stop(
                state, record, deliberation, tier, invocations, started, action
            )

        seat = route(action.action_type)
        performer = self.roles.get(seat)
        if performer is None:
            raise MissingRoleError(
                f"action {action.action_type} routes to {seat}, but no role "
                f"is registered for that seat"
            )

        # Budget gate: work that cannot be paid for is never started.
        estimated = selected.valuation.expected_cost
        if not state.budget.can_afford(estimated):
            return self._halted(
                state, record, deliberation, tier, invocations, started,
                f"insufficient budget for {action.action_type}: the estimated "
                f"cost exceeds the remaining budget",
            )

        attempt = ActionAttempt(action=action).started()
        state = state.begin_attempt(attempt)
        invocation = RoleInvocation(
            role=seat,
            assignment=action,
            context=_context_for(state, self.store, action),
            allowed_actions=frozenset({action.action_type}),
            expected_output=expected_proposals(action.action_type),
            budget=estimated,
        )

        failures = 0
        validation: tuple[ValidationReport, ...] = ()
        results: tuple[ExperimentResult, ...] = ()
        try:
            proposals = performer.perform(invocation)
            invocations += 1
            _check_contract(invocation, proposals, seat)
            results = tuple(
                p.result for p in proposals if isinstance(p, ResultProposal)
            )
            # The deterministic gate: nothing enters authoritative scientific
            # state unless code has checked it. Raises before any commit.
            validation = _gate_results(state, action, proposals, results)
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.SUCCEEDED,
                    produced=tuple(
                        pid for p in proposals for pid in payload_ids(p)
                    ),
                    actual_cost=_actual_cost(results, estimated),
                ),
                proposals=proposals,
            )
        except ValidationGateError as exc:
            failures = 1
            validation = exc.reports
            results = ()
            step_notes.append(
                f"engineering failure: {action.action_type} rejected by the "
                f"deterministic validation gate — {exc} (run outputs preserved)"
            )
            bundle = _failed_bundle(attempt.id, str(exc), estimated)
        except TransitionError as exc:
            failures = 1
            results = ()
            step_notes.append(
                f"engineering failure: {action.action_type} — {exc}"
            )
            bundle = _failed_bundle(attempt.id, str(exc), estimated)
        except Exception as exc:  # a role failing is an outcome, not a crash
            failures = 1
            invocations += 1  # the invocation happened even though it failed
            results = ()
            step_notes.append(
                f"engineering failure: {seat} raised during "
                f"{action.action_type} — {exc}"
            )
            bundle = _failed_bundle(attempt.id, str(exc), estimated)

        contradictions_before = len(find_contradictions(state))
        try:
            state = commit_bundle(state, bundle, self.store)
        except TransitionError as exc:
            failures += 1
            results = ()
            step_notes.append(f"engineering failure: commit rejected — {exc}")
            state = commit_bundle(
                state, _failed_bundle(attempt.id, str(exc), estimated), self.store
            )
        outcome = _outcome_of(state, attempt.id)

        # -- Tier 0 aftermath of committed results ---------------------------
        critic_reasons: tuple[str, ...] = ()
        critic_invoked = False
        if outcome.status is AttemptStatus.SUCCEEDED and results:
            completed = tuple(r for r in results if r.succeeded)
            self._results_since_synthesis += len(completed)
            for result in completed:
                state = self._transcribe(state, result)
                reasons = self._critic_reasons(state, result)
                if reasons and not critic_reasons:
                    critic_reasons = reasons
                    if self.config.critic_enabled:
                        state, invoked = self._invoke_critic(
                            state, result, reasons
                        )
                        invocations += invoked
                        critic_invoked = invoked > 0
            step_notes.extend(
                self._execution_failure_notes(state, results)
            )

        state = state.apply(action)
        state, overrun_note, exhausted = _reconcile_cost(
            state, outcome.actual_cost, estimated
        )
        if overrun_note is not None:
            step_notes.append(overrun_note)
        if exhausted:
            return self._finish(
                state, record, deliberation, tier, invocations, started,
                attempt_id=attempt.id, outcome=outcome, seat=seat,
                validation=validation, critic_reasons=critic_reasons,
                critic_invoked=critic_invoked, failures=failures,
                results=results, notes=tuple(step_notes),
                halt_reason="budget exhausted after cost overrun",
            )

        synthesis = self._maybe_synthesize(
            state,
            new_contradiction=len(find_contradictions(state))
            > contradictions_before,
            stopping=False,
        )
        if synthesis is not None:
            invocations += 1

        self._notes = (*self._notes, *step_notes)
        return self._finish(
            state, record, deliberation, tier, invocations, started,
            attempt_id=attempt.id, outcome=outcome, seat=seat,
            validation=validation, critic_reasons=critic_reasons,
            critic_invoked=critic_invoked, failures=failures,
            results=results, synthesis=synthesis, notes=tuple(step_notes),
            halt_reason=None,
        )

    # -- deterministic aftermath --------------------------------------------

    def _transcribe(
        self, state: ResearchState, result: ExperimentResult
    ) -> ResearchState:
        """Read one gate-validated, committed result into evidence. Zero
        model calls: the reading reuses the mechanical prediction test the
        commit already produced, and the proposal is attributed to the
        runtime."""
        spec = state.experiment(result.spec_id)
        assert spec is not None  # the bundle could not have committed otherwise
        prediction = state.prediction(spec.prediction_id)
        test = (
            state.test_for_result(prediction.id, result.id)
            if prediction is not None
            else None
        )
        evidence = evidence_from_result(result, test=test)
        return commit(
            state, EvidenceProposal(evidence=evidence, proposer=_READER), self.store
        )

    def _execution_failure_notes(
        self, state: ResearchState, results: tuple[ExperimentResult, ...]
    ) -> list[str]:
        """Deterministic repeated-failure signal, counted from the execution
        record itself: failed/cancelled results of one experiment."""
        notes: list[str] = []
        for spec_id in {r.spec_id for r in results}:
            failed = sum(
                1
                for ref in state.results_for(spec_id)
                if not self.store.get_result(ref.result_id).succeeded
            )
            if failed >= self.config.repeated_failure_threshold:
                notes.append(
                    f"engineering: {failed} failed execution(s) of experiment "
                    f"{spec_id} — debug the implementation before rerunning"
                )
        return notes

    def _critic_reasons(
        self, state: ResearchState, result: ExperimentResult
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
        return self.critic_trigger.reasons(state, test=test)

    def _invoke_critic(
        self,
        state: ResearchState,
        result: ExperimentResult,
        reasons: tuple[str, ...],
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
            context=_critic_context(state, self.store, result, reasons),
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
            *self._notes,
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
        invocations: int,
        started: float,
        action: ResearchAction,
    ) -> StepReport:
        synthesis = self._maybe_synthesize(
            state, new_contradiction=False, stopping=True
        )
        if synthesis is not None:
            invocations += 1
        state = state.apply(action)
        return self._finish(
            state, record, deliberation, tier, invocations, started,
            attempt_id=None, outcome=None, seat=None,
            validation=(), critic_reasons=(), critic_invoked=False,
            failures=0, results=(), synthesis=synthesis, notes=(),
            halt_reason=action.rationale,
        )

    def _halted(
        self,
        state: ResearchState,
        record: DecisionRecord,
        deliberation: Deliberation,
        tier: ReasoningTier,
        invocations: int,
        started: float,
        reason: str,
    ) -> StepReport:
        return self._finish(
            state, record, deliberation, tier, invocations, started,
            attempt_id=None, outcome=None, seat=None,
            validation=(), critic_reasons=(), critic_invoked=False,
            failures=0, results=(), notes=(), halt_reason=reason,
        )

    def _finish(
        self,
        state: ResearchState,
        record: DecisionRecord,
        deliberation: Deliberation,
        tier: ReasoningTier,
        invocations: int,
        started: float,
        *,
        attempt_id: str | None,
        outcome: ActionOutcome | None,
        seat: RoleName | None,
        validation: tuple[ValidationReport, ...],
        critic_reasons: tuple[str, ...],
        critic_invoked: bool,
        failures: int,
        results: tuple[ExperimentResult, ...],
        notes: tuple[str, ...],
        synthesis: SynthesisReview | None = None,
        halt_reason: str | None,
    ) -> StepReport:
        self._persist(state)
        usage = self.usage.drain() if self.usage is not None else NO_USAGE
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
                    reasoning_invocations=invocations,
                    provider_calls=usage.calls,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    model=usage.model,
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
                    notes=(
                        *notes,
                        *(
                            f"validation:{check.name}"
                            for report in validation
                            for check in report.failures
                        ),
                    ),
                )
            )
        return StepReport(
            record=completed,
            state=state,
            deliberation=deliberation,
            tier=tier,
            reasoning_invocations=invocations,
            provider_usage=usage,
            validation=validation,
            critic_reasons=critic_reasons,
            critic_invoked=critic_invoked,
            synthesis=synthesis,
            notes=notes,
            halt_reason=halt_reason,
        )

    def _persist(self, state: ResearchState) -> None:
        if self.states is not None:
            self.states.persist(state)


# -- the deterministic gate ---------------------------------------------------


def _check_contract(
    invocation: RoleInvocation,
    proposals: tuple[Proposal, ...],
    seat: RoleName,
) -> None:
    rejected = [p for p in proposals if not invocation.permits(p)]
    if rejected:
        raise TransitionError(
            f"role {seat} returned proposal kind(s) outside its output "
            f"contract: {', '.join(type(p).__name__ for p in rejected)}"
        )


def _gate_results(
    state: ResearchState,
    action: ResearchAction,
    proposals: tuple[Proposal, ...],
    results: tuple[ExperimentResult, ...],
) -> tuple[ValidationReport, ...]:
    """Deterministically validate every returned result before any commit.

    Raises :class:`ValidationGateError` when a *completed* result fails any
    machine check — such a result must never enter authoritative scientific
    state, and no model is ever asked to overrule this. Failed/cancelled
    executions pass through: they commit as execution-failure records with
    inconclusive standing, which is honest.
    """
    if action.action_type in _SINGLE_RESULT_ACTIONS and len(results) != 1:
        raise ValidationGateError(
            f"{action.action_type} must return exactly one result, "
            f"got {len(results)}",
            reports=(),
        )

    reports: list[ValidationReport] = []
    for result in results:
        if not result.succeeded:
            continue  # an execution failure is a record, not a claim
        spec = _spec_for(state, proposals, result.spec_id)
        if spec is None:
            raise ValidationGateError(
                f"result {result.id} names unknown experiment {result.spec_id}",
                reports=tuple(reports),
            )
        prediction = state.prediction(spec.prediction_id)
        checks = list(
            validate_result(spec, result, prediction=prediction).checks
        )
        checks.append(_matches_assignment(action, result))
        checks.append(verify_artifact_integrity(result))
        report = ValidationReport(checks=tuple(checks))
        reports.append(report)
        if not report.passed:
            failed = ", ".join(check.name for check in report.failures)
            raise ValidationGateError(
                f"result {result.id} failed deterministic validation "
                f"({failed})",
                reports=tuple(reports),
            )
    return tuple(reports)


def _matches_assignment(
    action: ResearchAction, result: ExperimentResult
) -> ValidationCheck:
    """The result must come from the experiment the director selected."""
    if action.action_type not in _SINGLE_RESULT_ACTIONS or not action.targets:
        return ValidationCheck(name="result_matches_assignment", passed=True)
    return ValidationCheck(
        name="result_matches_assignment",
        passed=result.spec_id == action.targets[0],
        detail=(
            f"assignment targets {action.targets[0]}, result ran "
            f"{result.spec_id}"
        ),
    )


def _spec_for(
    state: ResearchState,
    proposals: tuple[Proposal, ...],
    spec_id: str,
) -> ExperimentSpec | None:
    """The spec a result claims — from the state, or proposed in the same
    bundle (a role may design and run in one assignment)."""
    spec = state.experiment(spec_id)
    if spec is not None:
        return spec
    return next(
        (
            p.spec
            for p in proposals
            if isinstance(p, ExperimentProposal) and p.spec.id == spec_id
        ),
        None,
    )


def _reconcile_cost(
    state: ResearchState,
    actual: ResourceCost,
    estimated: ResourceCost,
) -> tuple[ResearchState, str | None, bool]:
    """Bill the work that just committed; never leave it unbilled.

    Returns ``(state, note, exhausted)``. Affordable costs charge in full;
    an overrun beyond the invocation's estimate is recorded explicitly; a
    cost the remaining budget cannot cover drains the budget to its floor,
    records the overrun, and signals a safe halt.
    """
    if state.budget.can_afford(actual):
        note = None
        if not _fits(actual, estimated):
            note = (
                "budget overrun: actual cost exceeded the invocation's "
                "estimated budget; charged in full"
            )
        return state.charge(actual), note, False
    charged = _clamp(actual, state.budget)
    note = (
        "budget overrun: actual cost exceeded the remaining budget; "
        "remainder drained and the program halted"
    )
    return state.charge(charged), note, True


def _fits(cost: ResourceCost, cap: ResourceCost) -> bool:
    return (
        cost.wall_clock_seconds <= cap.wall_clock_seconds
        and cost.gpu_hours <= cap.gpu_hours
        and cost.usd <= cap.usd
        and cost.model_tokens <= cap.model_tokens
    )


def _clamp(cost: ResourceCost, budget: ResearchBudget) -> ResourceCost:
    """The largest affordable share of ``cost`` — what an overrun can still
    be billed against a nearly-empty budget."""
    return ResourceCost(
        wall_clock_seconds=min(cost.wall_clock_seconds, budget.wall_clock_seconds),
        gpu_hours=min(cost.gpu_hours, budget.gpu_hours),
        usd=min(cost.usd, budget.usd),
        model_tokens=min(cost.model_tokens, budget.model_tokens),
    )


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
        notes=reasons,
    )
