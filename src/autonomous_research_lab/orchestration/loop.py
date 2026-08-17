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
      -> cost reconciled against the budget — work is never committed
         unbilled, and rejected work is billed at its actual known cost,
         never silently at the estimate
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
  preserved — and the execution's actual cost and runtime stay on the
  operational record and are billed — and the director sees a deterministic
  note next step — never a critic, because no model opinion can override
  arithmetic;
* a failed/cancelled *execution* is an honest execution record: it commits
  with inconclusive scientific standing, is deterministically diagnosed
  (``execution.failure_classifier``), and — when a debugger is wired in —
  enters the **bounded repair loop**, each retry a new auditable, billed
  attempt; repeated failures of one experiment raise a deterministic
  engineering note;
* a **methodologically rejected** design never executes: the response is
  *redesign the experiment*, not debugging and not a scientific negative;
* an **analytically invalid** judgment (post-hoc run selection) raises a
  *redo the analysis* note without blaming the executions it ignored;
* a *scientifically valid* consequential result — contradiction, challenged
  standing, large effect — is what earns a critic.

Validity and outcome stay orthogonal throughout (``runtime.verification``):
a completed run's outcome is never routed to debugging for being
disappointing, and a conclusive negative is promoted to verified scientific
evidence only when execution, implementation, methodology and analysis are
all positively resolved — otherwise the observation is preserved with its
validity explicitly unresolved.
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
from ..core.prediction import Consistency
from ..core.proposals import (
    EvidenceProposal,
    ExperimentProposal,
    Proposal,
    ResultProposal,
    payload_ids,
)
from ..core.state import ResearchState
from ..evidence.store import EvidenceStore
from ..execution.failure_classifier import diagnose_failure
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
from ..runtime.preflight import PreflightError
from ..runtime.validation import (
    ValidationCheck,
    ValidationReport,
    evidence_from_result,
    validate_result,
    verify_artifact_integrity,
)
from ..runtime.verification import (
    CheckState,
    ControlSource,
    ImplementationVerifier,
    MethodologyReviewer,
    OutcomeStanding,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
    derive_validity,
    evaluate_controls,
    outcome_standing,
    verify_analysis_coverage,
)
from .critic_trigger import CriticTrigger
from .debug_loop import DebugAttempt, ExperimentDebugger, is_debuggable
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


@dataclass
class _StepStats:
    """Verification/debugging accounting accumulated during one step and
    flushed into :class:`StepMetrics` and :class:`StepReport` at the end."""

    failure_category: str = ""
    debug_attempts: int = 0
    debug_resolved: bool = False
    verification: tuple[VerificationReport, ...] = ()
    verification_status: str = ""
    preflight_failed: bool = False
    control_failures: int = 0
    methodology_rejected: bool = False
    implementation_rejected: bool = False
    analysis_rejected: bool = False
    negative_result_verdict: str = ""


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
    verification: tuple[VerificationReport, ...] = ()
    """Verification reports of the completed results this step committed
    (debug-loop recoveries included) — the validity record, orthogonal to
    whatever the results' metrics say."""

    debug_attempts: int = 0
    debug_resolved: bool = False
    notes: tuple[str, ...] = ()
    """Deterministic runtime notes raised this step (engineering failures,
    repeated execution failures, budget overruns, validity findings)."""

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

    debugger: ExperimentDebugger | None = None
    """The bounded repair loop for failed executions. ``None`` (and
    ``config.debug_enabled = False``) leaves failures diagnosed and noted
    but never automatically repaired — the removable-component seam."""

    methodology_reviewer: MethodologyReviewer | None = None
    """Reviews each design once before its first execution, when
    ``config.methodology_review_enabled``. A rejection means redesign."""

    implementation_verifier: ImplementationVerifier | None = None
    """Event-triggered faithfulness check (failed controls, uncovered
    conclusive negatives), when ``config.implementation_verification_enabled``."""

    control_source: ControlSource | None = None
    """Experiment-specific positive controls, looked up per spec when
    ``config.positive_controls_enabled``."""

    _results_since_synthesis: int = field(default=0, init=False)
    _notes: tuple[str, ...] = field(default=(), init=False)
    _methodology_checks: dict[str, VerificationCheck] = field(
        default_factory=dict, init=False
    )
    """Per-spec methodology verdicts: each design is reviewed at most once,
    which is what keeps the review selective rather than per-run."""

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
        stats = _StepStats()
        assessments_before = {a.id for a in state.assessments}
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

        # Methodology gate: reviewed once per design, before its first —
        # potentially expensive — execution. Rejection means redesign; the
        # code is never debugged for a flaw in the science.
        if action.action_type in _SINGLE_RESULT_ACTIONS and action.targets:
            methodology, reviewed = self._methodology_check(
                state, action.targets[0]
            )
            invocations += reviewed
            if methodology is not None and methodology.state is CheckState.FAIL:
                return self._reject_methodology(
                    state, record, deliberation, tier, invocations, started,
                    action=action, check=methodology, stats=stats,
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
        # Operational vs scientific record, kept deliberately separate:
        # ``executed_results`` is every result the role actually returned —
        # retained for cost, timing and diagnostics even when nothing
        # commits; ``committed_results`` is the gate-approved subset that
        # entered authoritative scientific state.
        executed_results: tuple[ExperimentResult, ...] = ()
        committed_results: tuple[ExperimentResult, ...] = ()

        # The role invocation happens exactly once, counted here. Only an
        # exception raised by ``perform`` itself is attributed to the role.
        invocations += 1
        proposals: tuple[Proposal, ...] = ()
        role_error: Exception | None = None
        try:
            proposals = performer.perform(invocation)
        except Exception as exc:  # a role failing is an outcome, not a crash
            role_error = exc

        if role_error is not None:
            failures = 1
            if isinstance(role_error, PreflightError):
                stats.preflight_failed = True
                stats.failure_category = "preflight"
                step_notes.append(
                    f"engineering failure: preflight rejected the job before "
                    f"execution — {role_error}; expensive execution prevented"
                )
                # Nothing launched: a prevented execution bills no compute.
                bundle = _failed_bundle(attempt.id, str(role_error), NO_COST)
            else:
                step_notes.append(
                    f"engineering failure: {seat} raised during "
                    f"{action.action_type} — {role_error}"
                )
                # No result came back, so the estimate is the best known cost.
                bundle = _failed_bundle(attempt.id, str(role_error), estimated)
        else:
            executed_results = tuple(
                p.result for p in proposals if isinstance(p, ResultProposal)
            )
            try:
                _check_contract(invocation, proposals, seat)
                # The deterministic gate: nothing enters authoritative
                # scientific state unless code has checked it. Raises before
                # any commit. Unexpected exceptions from the checks
                # themselves propagate — mislabeling an orchestration bug as
                # a role failure would corrupt the record.
                validation = _gate_results(
                    state, action, proposals, executed_results
                )
                bundle = CommitBundle(
                    attempt_id=attempt.id,
                    outcome=ActionOutcome(
                        status=AttemptStatus.SUCCEEDED,
                        produced=tuple(
                            pid for p in proposals for pid in payload_ids(p)
                        ),
                        actual_cost=_actual_cost(executed_results, estimated),
                    ),
                    proposals=proposals,
                )
            except ValidationGateError as exc:
                failures = 1
                validation = exc.reports
                step_notes.append(
                    f"engineering failure: {action.action_type} rejected by "
                    f"the deterministic validation gate — {exc} (run outputs "
                    f"preserved)"
                )
                bundle = _failed_bundle(
                    attempt.id,
                    str(exc),
                    _actual_cost(executed_results, estimated),
                )
            except TransitionError as exc:
                failures = 1
                step_notes.append(
                    f"engineering failure: {action.action_type} — {exc}"
                )
                bundle = _failed_bundle(
                    attempt.id,
                    str(exc),
                    _actual_cost(executed_results, estimated),
                )

        contradictions_before = len(find_contradictions(state))
        try:
            state = commit_bundle(state, bundle, self.store)
            if bundle.outcome.status is AttemptStatus.SUCCEEDED:
                committed_results = executed_results
        except TransitionError as exc:
            failures += 1
            step_notes.append(f"engineering failure: commit rejected — {exc}")
            state = commit_bundle(
                state,
                _failed_bundle(
                    attempt.id,
                    str(exc),
                    _actual_cost(executed_results, estimated),
                ),
                self.store,
            )
        outcome = _outcome_of(state, attempt.id)

        # -- Tier 0 aftermath of committed results ---------------------------
        critic_reasons: tuple[str, ...] = ()
        critic_invoked = False
        if outcome.status is AttemptStatus.SUCCEEDED and committed_results:
            completed = tuple(r for r in committed_results if r.succeeded)
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
                verification_notes, verify_invocations = self._verify_result(
                    state, result, stats
                )
                step_notes.extend(verification_notes)
                invocations += verify_invocations
            step_notes.extend(
                self._execution_failure_notes(state, committed_results)
            )

            # A committed failed execution enters the bounded repair loop —
            # entry is by failure diagnosis, never by scientific outcome.
            failed_runs = tuple(r for r in committed_results if not r.succeeded)
            if failed_runs:
                state, debug_invocations, debug_results, exhausted = (
                    self._handle_failed_execution(
                        state, failed_runs[0], stats, step_notes
                    )
                )
                invocations += debug_invocations
                executed_results = (*executed_results, *debug_results)
                if exhausted:
                    return self._finish(
                        state, record, deliberation, tier, invocations, started,
                        attempt_id=attempt.id, outcome=outcome, seat=seat,
                        validation=validation, critic_reasons=critic_reasons,
                        critic_invoked=critic_invoked, failures=failures,
                        executed_results=executed_results,
                        notes=tuple(step_notes), stats=stats,
                        halt_reason="budget exhausted during debugging",
                    )

        step_notes.extend(self._analysis_notes(assessments_before, state, stats))

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
                executed_results=executed_results, notes=tuple(step_notes),
                stats=stats,
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
            executed_results=executed_results, synthesis=synthesis,
            notes=tuple(step_notes), stats=stats,
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

    # -- experiment validity -------------------------------------------------

    def _methodology_check(
        self, state: ResearchState, spec_id: str
    ) -> tuple[VerificationCheck | None, int]:
        """The one-time methodology review of a design, cached per spec —
        that cache is what keeps the review selective rather than per-run.
        Returns the verdict (``None`` when review is off or unwired) and
        the reasoning invocations spent (1 on a cache miss, else 0)."""
        if (
            not self.config.methodology_review_enabled
            or self.methodology_reviewer is None
        ):
            return None, 0
        cached = self._methodology_checks.get(spec_id)
        if cached is not None:
            return cached, 0
        spec = state.experiment(spec_id)
        if spec is None:
            return None, 0
        check = self.methodology_reviewer.review(
            spec, state.prediction(spec.prediction_id), objective=state.objective
        )
        self._methodology_checks[spec_id] = check
        return check, 1

    def _verify_result(
        self, state: ResearchState, result: ExperimentResult, stats: _StepStats
    ) -> tuple[list[str], int]:
        """Assemble the validity record of one committed completed result.

        Deterministic checks come first and are never overridable; the
        semantic verifier is consulted only when a deterministic signal (a
        failed or uncertain control) or an uncovered conclusive negative
        justifies the spend. Returns the notes raised and the reasoning
        invocations added.
        """
        spec = state.experiment(result.spec_id)
        assert spec is not None  # the result could not have committed otherwise
        prediction = state.prediction(spec.prediction_id)
        test = (
            state.test_for_result(prediction.id, result.id)
            if prediction is not None
            else None
        )
        notes: list[str] = []
        invocations = 0
        checks: list[VerificationCheck] = [
            VerificationCheck(
                dimension=ValidityDimension.EXECUTION,
                name="deterministic_validation",
                state=CheckState.PASS,
                detail=(
                    "process completed and passed the pre-commit gate "
                    "(declared metrics, finite values, seed, artifact "
                    "integrity)"
                ),
            )
        ]

        control_checks: tuple[VerificationCheck, ...] = ()
        if self.config.positive_controls_enabled and self.control_source is not None:
            control_checks = evaluate_controls(
                self.control_source(spec), result.metrics
            )
            checks.extend(control_checks)
            failed_controls = tuple(
                c for c in control_checks if c.state is CheckState.FAIL
            )
            stats.control_failures += len(failed_controls)
            for check in failed_controls:
                notes.append(
                    f"implementation uncertain: {check.name} failed "
                    f"({check.detail}) — verify the implementation; this is "
                    f"not a scientific negative"
                )

        negative = (
            test is not None and test.consistency is Consistency.INCONSISTENT
        )
        unresolved_controls = any(
            c.state in {CheckState.FAIL, CheckState.UNCERTAIN}
            for c in control_checks
        )
        if (
            self.config.implementation_verification_enabled
            and self.implementation_verifier is not None
            and (unresolved_controls or (negative and not control_checks))
        ):
            check = self.implementation_verifier.verify(
                spec, result, prediction, tuple(checks)
            )
            checks.append(check)
            invocations += 1
            if check.state is CheckState.FAIL:
                stats.implementation_rejected = True
                notes.append(
                    f"implementation rejected: {check.detail} — debug or "
                    f"reimplement; the outcome is not scientific evidence"
                )
        if not any(
            c.dimension is ValidityDimension.IMPLEMENTATION for c in checks
        ):
            checks.append(
                VerificationCheck(
                    dimension=ValidityDimension.IMPLEMENTATION,
                    name="implementation_faithfulness",
                    state=CheckState.UNCERTAIN,
                    detail="no positive controls or verifier consulted",
                )
            )

        methodology = self._methodology_checks.get(spec.id)
        checks.append(
            methodology
            if methodology is not None
            else VerificationCheck(
                dimension=ValidityDimension.METHODOLOGY,
                name="methodological_validity",
                state=CheckState.UNCERTAIN,
                detail="no methodology review performed",
            )
        )
        checks.append(
            VerificationCheck(
                dimension=ValidityDimension.ANALYSIS,
                name="raw_result_reading",
                state=CheckState.PASS,
                detail=(
                    "outcome read by the pre-registered mechanical "
                    "prediction check; no downstream aggregation involved"
                ),
            )
        )

        report = VerificationReport(checks=tuple(checks))
        validity = derive_validity(report)
        stats.verification = (*stats.verification, report)
        stats.verification_status = validity.value

        if negative:
            if outcome_standing(validity) is OutcomeStanding.VERIFIED_EVIDENCE:
                stats.negative_result_verdict = "accepted"
                notes.append(
                    f"verified scientific negative: result {result.id} "
                    f"refutes its prediction with every validity dimension "
                    f"resolved — preserved as evidence, not a debugging "
                    f"matter"
                )
            else:
                stats.negative_result_verdict = "deferred"
                if self._verification_wired():
                    notes.append(
                        f"negative outcome observed but validity unresolved "
                        f"({validity}): observation preserved without "
                        f"promotion to scientific evidence — and not routed "
                        f"to debugging for being negative"
                    )
        return notes, invocations

    def _verification_wired(self) -> bool:
        return (
            self.control_source is not None
            or self.implementation_verifier is not None
            or self.methodology_reviewer is not None
        )

    def _handle_failed_execution(
        self,
        state: ResearchState,
        result: ExperimentResult,
        stats: _StepStats,
        step_notes: list[str],
    ) -> tuple[ResearchState, int, tuple[ExperimentResult, ...], bool]:
        """Diagnose one committed failed execution and, when enabled and
        wired, run the bounded repair loop. Returns the state, the
        reasoning invocations added, every rerun result (for runtime
        accounting), and whether the budget was exhausted."""
        diagnosis = diagnose_failure(result)
        stats.failure_category = diagnosis.category.value
        step_notes.append(
            f"engineering failure diagnosed: {diagnosis.category} "
            f"({diagnosis.repairability}) — {diagnosis.rationale}"
        )
        if (
            not self.config.debug_enabled
            or self.debugger is None
            or not is_debuggable(diagnosis)
        ):
            return state, 0, (), False
        spec = state.experiment(result.spec_id)
        assert spec is not None  # committed results always name a known spec

        invocations = 0
        reruns: list[ExperimentResult] = []
        current = result
        for number in range(1, self.config.max_debug_attempts + 1):
            if not state.budget.can_afford(spec.estimated_cost):
                step_notes.append(
                    f"debugging stopped before attempt {number}: "
                    f"insufficient budget"
                )
                return state, invocations, tuple(reruns), False
            session = self.debugger.debug(spec, current, max_attempts=1)
            if not session.attempts:
                step_notes.append(f"debugging stopped: {session.stop_reason}")
                return state, invocations, tuple(reruns), False
            (attempt_record,) = session.attempts
            invocations += 1  # the repair proposal is reasoning-seat work
            stats.debug_attempts += 1
            retry = attempt_record.result
            reruns.append(retry)
            state, committed, exhausted = self._commit_debug_attempt(
                state, spec, attempt_record, number, step_notes
            )
            if exhausted:
                return state, invocations, tuple(reruns), True
            if retry.succeeded and committed:
                stats.debug_resolved = True
                step_notes.append(
                    f"debugging succeeded on attempt {number}: a valid "
                    f"execution of {spec.id} was recovered — its scientific "
                    f"outcome stands on its own"
                )
                state = self._transcribe(state, retry)
                notes, verify_invocations = self._verify_result(
                    state, retry, stats
                )
                step_notes.extend(notes)
                invocations += verify_invocations
                return state, invocations, tuple(reruns), False
            if retry.succeeded and not committed:
                # Completed but gate-rejected: not re-diagnosable as an
                # engineering failure of the process; stop and surface it.
                return state, invocations, tuple(reruns), False
            current = retry
        step_notes.append(
            f"debugging stopped after {self.config.max_debug_attempts} "
            f"attempt(s) without a valid execution of {spec.id}"
        )
        return state, invocations, tuple(reruns), False

    def _commit_debug_attempt(
        self,
        state: ResearchState,
        spec: ExperimentSpec,
        record: DebugAttempt,
        number: int,
        step_notes: list[str],
    ) -> tuple[ResearchState, bool, bool]:
        """Commit one repair rerun as its own auditable, billed attempt.
        Returns the state, whether the rerun committed cleanly, and whether
        the budget was exhausted paying for it."""
        action = ResearchAction(
            action_type=ResearchActionType.DEBUG,
            rationale=(
                f"repair attempt {number} for {spec.id}: "
                f"{record.diagnosis.category} — {record.repair_rationale}"
            ),
            targets=(spec.id,),
        )
        attempt = ActionAttempt(action=action).started()
        state = state.begin_attempt(attempt)
        result = record.result
        proposal = ResultProposal(result=result, proposer="runtime:debug-loop:v1")
        committed = True
        try:
            if result.spec_id != spec.id:
                raise ValidationGateError(
                    f"repair rerun reports spec {result.spec_id}, "
                    f"not {spec.id}",
                    reports=(),
                )
            _gate_results(state, action, (proposal,), (result,))
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.SUCCEEDED,
                    produced=payload_ids(proposal),
                    actual_cost=result.cost,
                ),
                proposals=(proposal,),
            )
        except ValidationGateError as exc:
            committed = False
            step_notes.append(
                f"engineering failure: debug rerun rejected by the "
                f"deterministic validation gate — {exc} (run outputs "
                f"preserved)"
            )
            bundle = _failed_bundle(attempt.id, str(exc), result.cost)
        state = commit_bundle(state, bundle, self.store)
        state = state.apply(action)
        estimated = (
            spec.estimated_cost if not spec.estimated_cost.is_zero else result.cost
        )
        state, overrun_note, exhausted = _reconcile_cost(
            state, result.cost, estimated
        )
        if overrun_note is not None:
            step_notes.append(overrun_note)
        return state, committed, exhausted

    def _analysis_notes(
        self,
        before: set[str],
        state: ResearchState,
        stats: _StepStats,
    ) -> list[str]:
        """Deterministic post-hoc-selection guard over every judgment this
        step added: an assessment citing only part of the conclusive
        evidence available to it is an analysis error. The response is
        *redo the analysis* — never rerun the valid experiments beneath it."""
        notes: list[str] = []
        for assessment in state.assessments:
            if assessment.id in before:
                continue
            hypothesis_id = _assessed_hypothesis(state, assessment.subject_id)
            if hypothesis_id is None:
                continue
            check = verify_analysis_coverage(
                cited_evidence_ids=assessment.evidence_ids,
                conclusive_evidence_ids=self._conclusive_evidence_ids(
                    state, hypothesis_id
                ),
            )
            if check.state is CheckState.FAIL:
                stats.analysis_rejected = True
                notes.append(
                    f"analytical failure: assessment {assessment.id} — "
                    f"{check.detail}; redo the analysis over the full result "
                    f"family (the underlying executions remain valid)"
                )
        return notes

    def _conclusive_evidence_ids(
        self, state: ResearchState, hypothesis_id: str
    ) -> tuple[str, ...]:
        conclusive_results = {
            test.result_id
            for prediction in state.predictions_for(hypothesis_id)
            for test in state.tests_for(prediction.id)
            if test.consistency is not Consistency.INCONCLUSIVE
        }
        return tuple(
            evidence_id
            for evidence_id in state.evidence_ids
            if self.store.get_evidence(evidence_id).result_id
            in conclusive_results
        )

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
            # The critic is under the same mechanical output contract as
            # every other seat: an unauthorized proposal kind rejects the
            # whole bundle and the critic attempt resolves as failed.
            _check_contract(invocation, proposals, RoleName.RESULT_ANALYST)
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

    def _reject_methodology(
        self,
        state: ResearchState,
        record: DecisionRecord,
        deliberation: Deliberation,
        tier: ReasoningTier,
        invocations: int,
        started: float,
        *,
        action: ResearchAction,
        check: VerificationCheck,
        stats: _StepStats,
    ) -> StepReport:
        """A design the methodology review rejected never executes. The
        response surfaced to the director is *redesign the experiment* —
        explicitly not debugging, and no scientific negative is recorded."""
        stats.methodology_rejected = True
        attempt = ActionAttempt(action=action).started()
        state = state.begin_attempt(attempt)
        detail = check.detail or "the design cannot answer the stated question"
        state = commit_bundle(
            state,
            _failed_bundle(
                attempt.id,
                f"methodologically invalid: {detail} — redesign the experiment",
                NO_COST,
            ),
            self.store,
        )
        outcome = _outcome_of(state, attempt.id)
        note = (
            f"methodological failure: experiment {action.targets[0]} rejected "
            f"before execution — redesign the experiment; do not debug the "
            f"code, and record no scientific negative ({detail})"
        )
        state = state.apply(action)
        self._notes = (*self._notes, note)
        return self._finish(
            state, record, deliberation, tier, invocations, started,
            attempt_id=attempt.id, outcome=outcome, seat=None,
            validation=(), critic_reasons=(), critic_invoked=False,
            failures=0, executed_results=(), notes=(note,), stats=stats,
            halt_reason=None,
        )

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
            failures=0, executed_results=(), synthesis=synthesis, notes=(),
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
            failures=0, executed_results=(), notes=(), halt_reason=reason,
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
        executed_results: tuple[ExperimentResult, ...],
        notes: tuple[str, ...],
        synthesis: SynthesisReview | None = None,
        stats: _StepStats | None = None,
        halt_reason: str | None,
    ) -> StepReport:
        stats = stats if stats is not None else _StepStats()
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
                    # Everything that actually ran, gate-rejected work
                    # included — runtime is spent whether or not it commits.
                    experiment_seconds=sum(
                        r.runtime_seconds for r in executed_results
                    ),
                    estimated_usd=(
                        selected.valuation.expected_cost.usd if selected else 0.0
                    ),
                    failures=failures,
                    critic_invoked=critic_invoked,
                    critic_reasons=critic_reasons,
                    synthesis_invoked=synthesis is not None,
                    failure_category=stats.failure_category,
                    debug_attempts=stats.debug_attempts,
                    debug_resolved=stats.debug_resolved,
                    verification_status=stats.verification_status,
                    preflight_failed=stats.preflight_failed,
                    control_failures=stats.control_failures,
                    methodology_rejected=stats.methodology_rejected,
                    implementation_rejected=stats.implementation_rejected,
                    analysis_rejected=stats.analysis_rejected,
                    negative_result_verdict=stats.negative_result_verdict,
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
            verification=stats.verification,
            debug_attempts=stats.debug_attempts,
            debug_resolved=stats.debug_resolved,
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

    Every result — failed and cancelled executions included — must name a
    known experiment, correspond to the selected assignment, and pass the
    executor contract's artifact-integrity check: identity and provenance
    are not optional just because a run failed. Only successful completed
    results additionally face the scientific-success checks (declared
    metrics, finite values, prediction metric, seed).

    Raises :class:`ValidationGateError` when any check fails — such a
    result must never enter authoritative scientific state, and no model is
    ever asked to overrule this. A correctly assigned failed execution
    passes and commits as an execution-failure record with inconclusive
    standing, which is honest.
    """
    if action.action_type in _SINGLE_RESULT_ACTIONS and len(results) != 1:
        raise ValidationGateError(
            f"{action.action_type} must return exactly one result, "
            f"got {len(results)}",
            reports=(),
        )

    reports: list[ValidationReport] = []
    for result in results:
        spec = _spec_for(state, proposals, result.spec_id)
        if spec is None:
            raise ValidationGateError(
                f"result {result.id} names unknown experiment {result.spec_id}",
                reports=tuple(reports),
            )
        checks: list[ValidationCheck] = []
        if result.succeeded:
            prediction = state.prediction(spec.prediction_id)
            checks.extend(
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


def _assessed_hypothesis(state: ResearchState, subject_id: str) -> str | None:
    """The hypothesis a new assessment bears on — directly, or through the
    claim it judges. ``None`` when the subject reaches no hypothesis."""
    if state.hypothesis(subject_id) is not None:
        return subject_id
    claim = state.claim(subject_id)
    return claim.hypothesis_id if claim is not None else None


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
