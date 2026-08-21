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
from ..core.assessment import AssessmentVerdict
from ..core.attempt import (
    ActionAttempt,
    ActionOutcome,
    AttemptPhase,
    AttemptStatus,
    SettlementBasis,
)
from ..core.budget import NO_COST, ResearchBudget, ResourceCost
from ..core.claim import EvidenceRelation
from ..core.commit import CommitBundle
from ..core.decision import DecisionRecord
from ..core.evidence import Evidence
from ..core.experiment import ExperimentResult, ExperimentSpec
from ..core.prediction import Consistency
from ..core.proposals import (
    AssessmentProposal,
    ClaimProposal,
    EvidenceProposal,
    ExperimentProposal,
    Proposal,
    ResultProposal,
    payload_ids,
)
from ..core.state import ResearchState, recorded_lineage, recording_lineage
from ..evidence.store import EvidenceStore, UnknownRecordError
from ..execution.executor import derive_job_id
from ..execution.failure_classifier import diagnose_failure
from ..persistence.commit_store import CommitBundleStore
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
from ..runtime.journal import AttemptJournal
from ..runtime.metrics import (
    NO_USAGE,
    MetricsSink,
    ProviderUsage,
    StepMetrics,
    UsageSource,
)
from ..runtime.playbook import Playbook, PlaybookAdvice
from ..runtime.preflight import TERMINAL_ENVIRONMENT_CHECKS, PreflightError
from ..runtime.providers import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
)
from ..runtime.spend import SpendLedger
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
    evaluate_controls,
    verify_analysis_coverage,
)
from ..runtime.verification_store import (
    InMemoryVerificationStore,
    ScientificAdmissibility,
    VerificationRecord,
    VerificationStore,
)
from .critic_trigger import CriticTrigger
from .debug_loop import (
    INVALID_IMPLEMENTATION,
    NO_FURTHER_FIX,
    ExperimentDebugger,
    ImplementationRepairTrigger,
    RepairProposal,
    is_debuggable,
)
from .director import Deliberation, FrontierDirector, deliberation_record
from .routing import expected_proposals, route
from .synthesis import SynthesisReview, SynthesisTrigger
from .trajectory import JsonlTrajectoryLogger
from .transitions import (
    TransitionError,
    commit,
    commit_bundle,
    store_facts,
)

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


class PromotionError(Exception):
    """A proposal tried to promote unverified observation into trusted
    scientific support. The observation itself is untouched — what is
    rejected is the *use*, not the record."""


class AnalysisValidityError(Exception):
    """A proposed judgment failed a deterministic analysis-validity check
    (e.g. post-hoc run selection). The response is *redo the analysis* —
    the executions underneath it remain valid and untouched."""


@dataclass
class _StepStats:
    """Verification/debugging accounting accumulated during one step and
    flushed into :class:`StepMetrics` and :class:`StepReport` at the end."""

    failure_category: str = ""
    debug_attempts: int = 0
    debug_resolved: bool = False
    implementation_debug_attempts: int = 0
    implementation_debug_resolved: bool = False
    verification: tuple[VerificationReport, ...] = ()
    verification_status: str = ""
    preflight_failed: bool = False
    control_failures: int = 0
    methodology_rejected: bool = False
    implementation_rejected: bool = False
    analysis_rejected: bool = False
    promotion_blocked: bool = False
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
    implementation_debug_attempts: int = 0
    implementation_debug_resolved: bool = False
    """Implementation repair is accounted separately from execution repair:
    the two entries share machinery but not preconditions, and the ablation
    needs to see them apart."""

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

    ledger: SpendLedger | None = None
    """The durable record of what this run spends, already bound to its
    run. Every charge posts one debit, keyed by the attempt that incurred
    it, and the ledger's balance must agree with the state's budget
    afterwards or the step fails loudly. ``None`` leaves spend on the
    state snapshots alone — the pre-existing behavior, kept as the
    explicit ablation."""

    journal: AttemptJournal | None = None
    """Where each attempt's phases are written down as they happen. With
    it, a process killed inside a step leaves a record saying how far it
    got; without it the run is recoverable between steps and not inside
    one — the pre-existing behavior, kept as the explicit ablation."""

    bundles: CommitBundleStore | None = None
    """Where the effect of an attempt is stored before it is applied. A
    bundle on disk is what lets a recovering process finish a step it
    never started, so this and ``journal`` are wired together or not at
    all."""

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

    verifications: VerificationStore = field(
        default_factory=InMemoryVerificationStore
    )
    """The durable verification record, keyed by result id. Written whenever
    a completed result is verified (i.e. whenever any verification component
    is wired); consulted by the scientific-promotion gate and projected into
    role contexts. Swap in a ``FileVerificationStore`` for persistence."""

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
        if (self.journal is None) != (self.bundles is None):
            raise ValueError(
                "a journal without a bundle store records phases nobody "
                "can act on, and a bundle store without a journal stores "
                "effects nobody will look for; wire both or neither"
            )
        if self.journal is not None and self.states is None:
            raise ValueError(
                "a journal names states a recovering process must be able "
                "to load; without a snapshot store there is nowhere to "
                "load them from"
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
        """One fast-loop iteration, with the slow loop run when due.

        Every state the iteration derives is persisted, oldest first, so
        the snapshot store holds a chain rather than a sequence of heads
        with unreachable parents. The head is written last: a crash
        leaves a shorter chain, never a state whose ancestry is missing.
        """
        with recording_lineage() as derived:
            report = self._step(state)
        for successor in derived:
            self._persist(successor)
        return report

    def _step(self, state: ResearchState) -> StepReport:
        started = time.monotonic()
        step_notes: list[str] = []
        stats = _StepStats()
        admissible = self._admissibility()
        frontier = build_frontier(
            state,
            recent_results=self.config.recent_results,
            open_decisions=self._notes,
            admissible=admissible,
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
        self._open_attempt(attempt.id, state, estimated)
        invocation = RoleInvocation(
            role=seat,
            assignment=action,
            context=_context_for(
                state, self.store, self.verifications, action, admissible
            ),
            allowed_actions=frozenset({action.action_type}),
            expected_output=expected_proposals(action.action_type),
            budget=estimated,
            attempt_id=attempt.id,
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

        terminal_halt: str | None = None
        if role_error is not None:
            failures = 1
            terminal_halt = _terminal_failure_reason(role_error)
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
                # The deterministic gates: nothing enters authoritative
                # scientific state unless code has checked it, and no
                # unverified observation is promoted into trusted support.
                # Both raise before any commit. Unexpected exceptions from
                # the checks themselves propagate — mislabeling an
                # orchestration bug as a role failure would corrupt the
                # record.
                self._gate_promotions(proposals)
                self._gate_analysis(state, proposals)
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
            except PromotionError as exc:
                failures = 1
                stats.promotion_blocked = True
                step_notes.append(
                    f"scientific promotion blocked: {exc} — the observation "
                    f"is preserved; resolve its validity (or cite it as "
                    f"inconclusive) before using it as support"
                )
                bundle = _failed_bundle(
                    attempt.id,
                    str(exc),
                    _actual_cost(executed_results, estimated),
                )
            except AnalysisValidityError as exc:
                failures = 1
                stats.analysis_rejected = True
                step_notes.append(
                    f"analytical failure: {exc}; redo the analysis over the "
                    f"full result family — the underlying executions remain "
                    f"valid and untouched"
                )
                bundle = _failed_bundle(
                    attempt.id,
                    str(exc),
                    _actual_cost(executed_results, estimated),
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

        contradictions_before = len(
            find_contradictions(state, admissible=admissible)
        )
        self._store_bundle(attempt.id, bundle)
        try:
            state = commit_bundle(state, bundle, self.store)
            if bundle.outcome.status is AttemptStatus.SUCCEEDED:
                committed_results = executed_results
        except TransitionError as exc:
            failures += 1
            step_notes.append(f"engineering failure: commit rejected — {exc}")
            fallback = _failed_bundle(
                attempt.id,
                str(exc),
                _actual_cost(executed_results, estimated),
            )
            self._store_bundle(attempt.id, fallback, replacing=True)
            state = commit_bundle(state, fallback, self.store)
        self._committed(attempt.id, state)
        outcome = _outcome_of(state, attempt.id)

        # -- Tier 0 aftermath of committed results ---------------------------
        critic_reasons: tuple[str, ...] = ()
        critic_invoked = False
        implementation_invalid: list[tuple[ExperimentResult, VerificationReport]] = []
        if outcome.status is AttemptStatus.SUCCEEDED and committed_results:
            completed = tuple(r for r in committed_results if r.succeeded)
            self._results_since_synthesis += len(completed)
            for result in completed:
                state = self._transcribe(state, result)
                # Verify before any reasoning consumes the result: the
                # durable record must exist by the time a critic (or any
                # later seat) could try to cite this result's evidence, or
                # the promotion gate would have nothing to enforce.
                verification_notes, verify_invocations, report_v = (
                    self._verify_result(state, result, stats)
                )
                step_notes.extend(verification_notes)
                invocations += verify_invocations
                if report_v is not None and _implementation_failures(report_v):
                    implementation_invalid.append((result, report_v))
                reasons = self._critic_reasons(state, result)
                if reasons and not critic_reasons:
                    critic_reasons = reasons
                    if self.config.critic_enabled:
                        state, invoked = self._invoke_critic(
                            state, result, reasons, stats, step_notes
                        )
                        invocations += invoked
                        critic_invoked = invoked > 0
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
                    state, _, _ = self._bill(
                        state, outcome.actual_cost, estimated,
                        charge_id=attempt.id,
                    )
                    return self._finish(
                        state, record, deliberation, tier, invocations, started,
                        attempt_id=attempt.id, outcome=outcome, seat=seat,
                        validation=validation, critic_reasons=critic_reasons,
                        critic_invoked=critic_invoked, failures=failures,
                        executed_results=executed_results,
                        notes=tuple(step_notes), stats=stats,
                        halt_reason="budget exhausted during debugging",
                    )

            # A completed run indicted by implementation-invalidity evidence
            # (failed control, verifier FAIL) enters bounded implementation
            # repair — the one path by which a *completed* run may be
            # repaired, and it is unreachable from a prediction test.
            if implementation_invalid:
                invalid_result, invalid_report = implementation_invalid[0]
                state, repair_invocations, repair_results, exhausted = (
                    self._handle_implementation_invalidity(
                        state, invalid_result, invalid_report, stats, step_notes
                    )
                )
                invocations += repair_invocations
                executed_results = (*executed_results, *repair_results)
                if exhausted:
                    state, _, _ = self._bill(
                        state, outcome.actual_cost, estimated,
                        charge_id=attempt.id,
                    )
                    return self._finish(
                        state, record, deliberation, tier, invocations, started,
                        attempt_id=attempt.id, outcome=outcome, seat=seat,
                        validation=validation, critic_reasons=critic_reasons,
                        critic_invoked=critic_invoked, failures=failures,
                        executed_results=executed_results,
                        notes=tuple(step_notes), stats=stats,
                        halt_reason="budget exhausted during debugging",
                    )

        state = state.apply(action)
        state, overrun_note, exhausted = self._bill(
            state, outcome.actual_cost, estimated, charge_id=attempt.id
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
                halt_reason="budget breached: the run spent past its grant",
            )

        synthesis = self._maybe_synthesize(
            state,
            new_contradiction=len(
                find_contradictions(state, admissible=admissible)
            )
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
            halt_reason=terminal_halt,
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
    ) -> tuple[list[str], int, VerificationReport | None]:
        """Assemble the validity record of one committed completed result
        and store it durably, keyed by result id.

        Deterministic checks come first and are never overridable; the
        semantic verifier is consulted only when a deterministic signal (a
        failed or uncertain control) or an uncovered conclusive negative
        justifies the spend. With no verification component wired at all
        (full ablation), no record is produced — absence of a record is the
        explicit marker of legacy, ungoverned results, and the promotion
        gate treats it as such. Returns the notes raised, the reasoning
        invocations added, and the report (``None`` when ablated).
        """
        if not self._verification_wired():
            return [], 0, None
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
        # The verdict becomes a durable record: this — not a step-local
        # note — is what the promotion gate and later steps consult.
        verdict = self.verifications.record(
            VerificationRecord(
                result_id=result.id, spec_id=result.spec_id, report=report
            )
        )
        stats.verification = (*stats.verification, report)
        stats.verification_status = verdict.validity.value

        if negative:
            if verdict.standing is OutcomeStanding.VERIFIED_EVIDENCE:
                stats.negative_result_verdict = "accepted"
                notes.append(
                    f"verified scientific negative: result {result.id} "
                    f"refutes its prediction with every validity dimension "
                    f"resolved — preserved as evidence, not a debugging "
                    f"matter"
                )
            else:
                stats.negative_result_verdict = "deferred"
                notes.append(
                    f"negative outcome observed but validity unresolved "
                    f"({verdict.validity}): observation preserved without "
                    f"promotion to scientific evidence — and not routed "
                    f"to debugging for being negative"
                )
        return notes, invocations, report

    def _verification_wired(self) -> bool:
        return (
            self.control_source is not None
            or self.implementation_verifier is not None
            or self.methodology_reviewer is not None
        )

    def _admissibility(self) -> ScientificAdmissibility:
        """The single scientific-admissibility policy every scientific
        consumer of this runtime shares — frontier projection,
        contradiction detection, critic triggering, analysis coverage."""
        return ScientificAdmissibility(
            verifications=self.verifications,
            governance_enabled=self.config.verification_governance_enabled,
        )

    # -- the scientific-promotion gate ---------------------------------------

    def _gate_promotions(self, proposals: tuple[Proposal, ...]) -> None:
        """Refuse to commit trusted scientific use of unverified observation.

        Under enabled verification governance the gate **fails closed**:
        trusted support (a SUPPORTS/CONTRADICTS link, a conclusive
        assessment) requires a durable ``VERIFIED_EVIDENCE`` record for the
        cited result, and a *missing* record blocks exactly like an adverse
        one — a lost store, a restart, or a mis-wired runtime must never
        silently restore trust. Legacy semantics exist only as explicit
        ablation (``verification_governance_enabled = False``), never as an
        inference from missing data. Inconclusive links and ``UNDETERMINED``
        assessments remain open to any observation — inspection is never
        blocked, promotion is. Raises :class:`PromotionError`;
        deterministic, no model consulted.
        """
        if not self.config.verification_governance_enabled:
            return  # explicit ablation: the deliberately ungoverned lab
        for proposal in proposals:
            match proposal:
                case ClaimProposal():
                    for link in proposal.links:
                        if link.relation is EvidenceRelation.INCONCLUSIVE:
                            continue
                        self._require_verified(
                            proposals,
                            link.evidence_id,
                            use=(
                                f"a {link.relation} link into claim "
                                f"{link.claim_id}"
                            ),
                        )
                case AssessmentProposal():
                    assessment = proposal.assessment
                    if assessment.verdict is AssessmentVerdict.UNDETERMINED:
                        continue
                    for evidence_id in assessment.evidence_ids:
                        self._require_verified(
                            proposals,
                            evidence_id,
                            use=(
                                f"a {assessment.verdict} assessment of "
                                f"{assessment.subject_id}"
                            ),
                        )
                case _:
                    continue

    def _require_verified(
        self, proposals: tuple[Proposal, ...], evidence_id: str, *, use: str
    ) -> None:
        evidence = self._evidence_named(proposals, evidence_id)
        if evidence is None:
            # Referential integrity is the transition layer's check, not
            # this gate's; a dangling reference is rejected there.
            return
        # The one canonical admissibility decision; the record is fetched
        # again only to render the precise reason for the rejection.
        if self._admissibility()(evidence.result_id):
            return
        verdict = self.verifications.get(evidence.result_id)
        if verdict is None:
            # Fail closed: no record is indistinguishable from a lost or
            # mis-wired store, so it is treated as UNVERIFIED, not trusted.
            raise PromotionError(
                f"evidence {evidence_id} rests on result "
                f"{evidence.result_id}, which has no verification record; "
                f"under enabled verification governance an unrecorded "
                f"result is treated as unverified and cannot serve as "
                f"{use} (disable governance explicitly to run ablated)"
            )
        raise PromotionError(
            f"evidence {evidence_id} rests on result "
            f"{evidence.result_id}, whose verification stands at "
            f"{verdict.validity} ({verdict.standing}); it may be "
            f"inspected but cannot serve as {use}"
        )

    def _evidence_named(
        self, proposals: tuple[Proposal, ...], evidence_id: str
    ) -> Evidence | None:
        """The cited evidence — proposed in this same bundle, or already in
        the store."""
        for proposal in proposals:
            if (
                isinstance(proposal, EvidenceProposal)
                and proposal.evidence.id == evidence_id
            ):
                return proposal.evidence
        try:
            return self.store.get_evidence(evidence_id)
        except UnknownRecordError:
            return None

    # -- the analysis-validity gate ------------------------------------------

    def _gate_analysis(
        self, state: ResearchState, proposals: tuple[Proposal, ...]
    ) -> None:
        """Deterministic analytical checks, applied **before** commit.

        A judgment that fails them — today, coverage: an assessment citing
        only part of the conclusive evidence available to its hypothesis
        (post-hoc run selection) — never enters authoritative scientific
        state. Raises :class:`AnalysisValidityError`; the response it
        surfaces is *redo the analysis*, never rerun the valid experiments
        beneath it.
        """
        for proposal in proposals:
            if not isinstance(proposal, AssessmentProposal):
                continue
            assessment = proposal.assessment
            hypothesis_id = _assessed_hypothesis(state, assessment.subject_id)
            if hypothesis_id is None:
                # The subject may be a claim proposed in this same bundle.
                hypothesis_id = next(
                    (
                        p.claim.hypothesis_id
                        for p in proposals
                        if isinstance(p, ClaimProposal)
                        and p.claim.id == assessment.subject_id
                    ),
                    None,
                )
            if hypothesis_id is None:
                continue
            # Coverage is owed to the scientifically admissible family
            # only: an invalid observation may be inspected and discussed,
            # but requiring it as trusted support would collide with the
            # promotion gate, which forbids citing it conclusively.
            check = verify_analysis_coverage(
                cited_evidence_ids=assessment.evidence_ids,
                conclusive_evidence_ids=_conclusive_evidence_ids(
                    state,
                    self.store,
                    hypothesis_id,
                    admissible=self._admissibility(),
                ),
            )
            if check.state is CheckState.FAIL:
                raise AnalysisValidityError(
                    f"assessment {assessment.id} of {assessment.subject_id} "
                    f"— {check.detail}"
                )

    def _handle_implementation_invalidity(
        self,
        state: ResearchState,
        result: ExperimentResult,
        report: VerificationReport,
        stats: _StepStats,
        step_notes: list[str],
    ) -> tuple[ResearchState, int, tuple[ExperimentResult, ...], bool]:
        """Bounded implementation repair of one completed-but-indicted run.

        Entry demands typed implementation-invalidity evidence (the failed
        checks become the :class:`ImplementationRepairTrigger`), never the
        scientific outcome. Within the one configured attempt bound, each
        iteration responds to the *latest* attempt's actual state:

        * a completed rerun earns its own fresh verification; a fresh
          implementation FAIL yields a **new trigger built from that run's
          report**, never a stale re-read of the original;
        * a rerun that crashes is an *execution* failure — it is diagnosed
          by the classifier and repaired with execution-repair semantics,
          not treated as another semantic implementation failure;
        * resolution means the newest run's implementation dimension no
          longer fails.

        The original result and its adverse verification record are
        preserved untouched, and every attempt on either path is a separate
        billed, auditable DEBUG occurrence.
        """
        if (
            not self.config.debug_enabled
            or self.debugger is None
            or self.debugger.implementation_strategy is None
        ):
            return state, 0, (), False
        trigger = ImplementationRepairTrigger(
            result_id=result.id, checks=_implementation_failures(report)
        )
        spec = state.experiment(result.spec_id)
        assert spec is not None  # committed results always name a known spec

        invocations = 0
        reruns: list[ExperimentResult] = []
        current = result
        for number in range(1, self.config.max_debug_attempts + 1):
            if not state.budget.can_afford(spec.estimated_cost):
                step_notes.append(
                    f"implementation repair stopped before attempt {number}: "
                    f"insufficient budget"
                )
                return state, invocations, tuple(reruns), False
            if current.succeeded:
                proposal = self.debugger.propose_reimplementation(
                    spec, current, trigger, number
                )
                basis = INVALID_IMPLEMENTATION
            else:
                # The previous reimplementation crashed: that is an
                # execution failure, diagnosed and repaired as one — while
                # the episode's single attempt bound keeps counting.
                diagnosis = diagnose_failure(current)
                step_notes.append(
                    f"reimplementation crashed — execution failure "
                    f"diagnosed: {diagnosis.category} "
                    f"({diagnosis.repairability}) — {diagnosis.rationale}"
                )
                if not is_debuggable(diagnosis):
                    step_notes.append(
                        "implementation repair stopped: the crashed rerun "
                        "is not diagnosable as a repairable execution "
                        "failure"
                    )
                    return state, invocations, tuple(reruns), False
                proposal = self.debugger.propose_repair(
                    spec, current, diagnosis, number
                )
                basis = str(diagnosis.category)
            if proposal is None:
                step_notes.append(
                    f"implementation repair stopped: {NO_FURTHER_FIX}"
                )
                return state, invocations, tuple(reruns), False
            invocations += 1  # the repair proposal is reasoning-seat work
            stats.implementation_debug_attempts += 1
            state, retry, committed, exhausted = self._repair_attempt(
                state,
                spec,
                proposal,
                repairing=current,
                basis=basis,
                number=number,
                step_notes=step_notes,
            )
            reruns.append(retry)
            if exhausted:
                return state, invocations, tuple(reruns), True
            if retry.succeeded and committed:
                state = self._transcribe(state, retry)
                # The rerun earns its own verification — nothing is
                # transferred from the run it replaces.
                notes, verify_invocations, retry_report = self._verify_result(
                    state, retry, stats
                )
                step_notes.extend(notes)
                invocations += verify_invocations
                if retry_report is None:
                    step_notes.append(
                        "implementation repair stopped: no verification is "
                        "available for the reimplementation, so its "
                        "implementation cannot be pronounced recovered"
                    )
                    return state, invocations, tuple(reruns), False
                fresh_failures = _implementation_failures(retry_report)
                if not fresh_failures:
                    stats.implementation_debug_resolved = True
                    step_notes.append(
                        f"implementation repair succeeded on attempt "
                        f"{number}: the reimplementation of {spec.id} no "
                        f"longer fails its implementation checks — its "
                        f"scientific outcome stands on its own"
                    )
                    return state, invocations, tuple(reruns), False
                # Still indicted: the next attempt answers THIS run's
                # evidence, not the original's.
                trigger = ImplementationRepairTrigger(
                    result_id=retry.id, checks=fresh_failures
                )
            if retry.succeeded and not committed:
                return state, invocations, tuple(reruns), False
            current = retry
        step_notes.append(
            f"implementation repair stopped after "
            f"{self.config.max_debug_attempts} attempt(s) without a valid "
            f"implementation of {spec.id}"
        )
        return state, invocations, tuple(reruns), False

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
        current_diagnosis = diagnosis
        for number in range(1, self.config.max_debug_attempts + 1):
            if not state.budget.can_afford(spec.estimated_cost):
                step_notes.append(
                    f"debugging stopped before attempt {number}: "
                    f"insufficient budget"
                )
                return state, invocations, tuple(reruns), False
            proposal = self.debugger.propose_repair(
                spec, current, current_diagnosis, number
            )
            if proposal is None:
                step_notes.append(f"debugging stopped: {NO_FURTHER_FIX}")
                return state, invocations, tuple(reruns), False
            invocations += 1  # the repair proposal is reasoning-seat work
            stats.debug_attempts += 1
            state, retry, committed, exhausted = self._repair_attempt(
                state,
                spec,
                proposal,
                repairing=current,
                basis=str(current_diagnosis.category),
                number=number,
                step_notes=step_notes,
            )
            reruns.append(retry)
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
                notes, verify_invocations, _ = self._verify_result(
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
            current_diagnosis = diagnose_failure(retry)
        step_notes.append(
            f"debugging stopped after {self.config.max_debug_attempts} "
            f"attempt(s) without a valid execution of {spec.id}"
        )
        return state, invocations, tuple(reruns), False

    def _repair_attempt(
        self,
        state: ResearchState,
        spec: ExperimentSpec,
        proposal: RepairProposal,
        *,
        repairing: ExperimentResult,
        basis: str,
        number: int,
        step_notes: list[str],
    ) -> tuple[ResearchState, ExperimentResult, bool, bool]:
        """Run one repair rerun as its own recoverable attempt, and commit
        it. Returns the state, what the rerun produced, whether it
        committed cleanly, and whether the budget was exhausted paying for
        it.

        The order here is the whole point of the method. The attempt is
        opened — snapshot, journal, reservation — *before* the job is
        submitted, so the rerun runs under an authorization of its own and
        its job id is on disk before the job exists. It used to be the
        other way round: the debugger submitted, and the attempt that
        answered for the job was opened once the result came back. That
        left a window in which a job could run with nothing anywhere
        recording that it had, and a process killed inside it lost the job
        outright — the spend was never charged to anything, and the
        outputs sat in the run directory with nothing on the record to
        find them by. Under a deterministic executor that cost seconds.
        Under a trainer it would cost the training.

        What the attempt is authorized to spend is the spec's estimate,
        or — where the design carries none — what the run being repaired
        actually cost, which is the closest thing to a forecast anyone
        has. Both are knowable *before* the job runs, which is the
        property the order above depends on: the old fallback was the
        rerun's own cost, and a number that only exists afterwards cannot
        authorize anything.
        """
        assert self.debugger is not None  # the callers hold one
        action = ResearchAction(
            action_type=ResearchActionType.DEBUG,
            rationale=(
                f"repair attempt {number} for {spec.id}: "
                f"{basis} — {proposal.rationale}"
            ),
            targets=(spec.id,),
        )
        attempt = ActionAttempt(action=action).started()
        estimated = (
            spec.estimated_cost
            if not spec.estimated_cost.is_zero
            else repairing.cost
        )
        state = state.begin_attempt(attempt)
        self._open_attempt(attempt.id, state, estimated)
        result = self.debugger.rerun(spec, proposal, attempt_id=attempt.id)
        produced = ResultProposal(
            result=result, proposer="runtime:debug-loop:v1"
        )
        committed = True
        try:
            if result.spec_id != spec.id:
                raise ValidationGateError(
                    f"repair rerun reports spec {result.spec_id}, "
                    f"not {spec.id}",
                    reports=(),
                )
            _gate_results(state, action, (produced,), (result,))
            bundle = CommitBundle(
                attempt_id=attempt.id,
                outcome=ActionOutcome(
                    status=AttemptStatus.SUCCEEDED,
                    produced=payload_ids(produced),
                    actual_cost=result.cost,
                ),
                proposals=(produced,),
            )
        except ValidationGateError as exc:
            committed = False
            step_notes.append(
                f"engineering failure: debug rerun rejected by the "
                f"deterministic validation gate — {exc} (run outputs "
                f"preserved)"
            )
            bundle = _failed_bundle(attempt.id, str(exc), result.cost)
        self._store_bundle(attempt.id, bundle)
        state = commit_bundle(state, bundle, self.store)
        self._committed(attempt.id, state)
        state = state.apply(action)
        state, overrun_note, exhausted = self._bill(
            state, result.cost, estimated, charge_id=attempt.id
        )
        if overrun_note is not None:
            step_notes.append(overrun_note)
        return state, result, committed, exhausted

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
        return self.critic_trigger.reasons(
            state, test=test, admissible=self._admissibility()
        )

    def _invoke_critic(
        self,
        state: ResearchState,
        result: ExperimentResult,
        reasons: tuple[str, ...],
        stats: _StepStats,
        step_notes: list[str],
    ) -> tuple[ResearchState, int]:
        critic = self.roles.get(RoleName.RESULT_ANALYST)
        if critic is None:
            return state, 0
        action = ResearchAction(
            action_type=ResearchActionType.ANALYZE,
            rationale="; ".join(reasons),
            targets=(result.id,),
        )
        # No separate authorization: a critic invocation is part of the
        # step that triggered it, and its cost is billed there.
        attempt = ActionAttempt(action=action).started()
        state = state.begin_attempt(attempt)
        invocation = RoleInvocation(
            role=RoleName.RESULT_ANALYST,
            assignment=action,
            context=_critic_context(
                state, self.store, self.verifications, result, reasons
            ),
            allowed_actions=frozenset({ResearchActionType.ANALYZE}),
            expected_output=expected_proposals(ResearchActionType.ANALYZE),
        )
        try:
            proposals = critic.perform(invocation)
            # The critic is under the same mechanical output contract and
            # the same promotion gate as every other seat: an unauthorized
            # proposal kind, or trusted use of unverified observation,
            # rejects the whole bundle and the critic attempt fails.
            _check_contract(invocation, proposals, RoleName.RESULT_ANALYST)
            self._gate_promotions(proposals)
            self._gate_analysis(state, proposals)
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
        except PromotionError as exc:
            stats.promotion_blocked = True
            step_notes.append(
                f"scientific promotion blocked: {exc} — the observation is "
                f"preserved; resolve its validity (or cite it as "
                f"inconclusive) before using it as support"
            )
            state = commit_bundle(
                state, _failed_bundle(attempt.id, str(exc), NO_COST), self.store
            )
        except AnalysisValidityError as exc:
            stats.analysis_rejected = True
            step_notes.append(
                f"analytical failure: {exc}; redo the analysis over the "
                f"full result family — the underlying executions remain "
                f"valid and untouched"
            )
            state = commit_bundle(
                state, _failed_bundle(attempt.id, str(exc), NO_COST), self.store
            )
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
            build_frontier(
                state,
                recent_results=self.config.recent_results,
                admissible=self._admissibility(),
            ),
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
                    implementation_debug_attempts=(
                        stats.implementation_debug_attempts
                    ),
                    implementation_debug_resolved=(
                        stats.implementation_debug_resolved
                    ),
                    verification_status=stats.verification_status,
                    preflight_failed=stats.preflight_failed,
                    control_failures=stats.control_failures,
                    methodology_rejected=stats.methodology_rejected,
                    implementation_rejected=stats.implementation_rejected,
                    analysis_rejected=stats.analysis_rejected,
                    promotion_blocked=stats.promotion_blocked,
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
            implementation_debug_attempts=stats.implementation_debug_attempts,
            implementation_debug_resolved=stats.implementation_debug_resolved,
            notes=notes,
            halt_reason=halt_reason,
        )

    # -- the attempt lifecycle -----------------------------------------------

    def _open_attempt(
        self, attempt_id: str, begun: ResearchState, estimated: ResourceCost
    ) -> None:
        """Make the attempt's starting point durable, write it down, and
        hold the money it is authorized to spend. In that order.

        The snapshot goes first because the journal is about to name it,
        and a record that names a state nobody stored is the one thing
        this whole mechanism must never write. It is the state with the
        attempt *begun* on it, which is what recovery has to apply a
        stored bundle to — a bundle names an attempt, and a state that
        never began that attempt refuses it.

        The journal goes before the reservation because money held for a
        reason no later process can reconstruct is worse than an attempt
        that plainly never got started.

        An attempt nobody expected to cost anything holds nothing and is
        not journalled: there is no authorization to record and nothing
        for recovery to answer for.
        """
        if estimated.is_zero:
            return
        if self.journal is not None:
            self._persist(begun)
            self.journal.record(
                attempt_id=attempt_id,
                phase=AttemptPhase.STARTED,
                state_id=begun.id,
                job_id=derive_job_id(attempt_id),
                reserved=estimated,
            )
        if self.ledger is not None and not self.ledger.holds(attempt_id):
            self.ledger.reserve(
                estimated,
                charge_id=attempt_id,
                reason=f"attempt {attempt_id}",
            )

    def _store_bundle(
        self, attempt_id: str, bundle: CommitBundle, *, replacing: bool = False
    ) -> None:
        """Put the whole effect of an attempt on disk before applying it.

        The facts go first. A bundle names its results and evidence rather
        than copying them, so a bundle stored while they are still in
        memory says the step can be finished from disk by a process that
        then cannot finish it — which is the one record this mechanism
        must never write. ``store_facts`` closes that window, and it is
        idempotent, so the commit below stores nothing a second time.

        ``replacing`` is the commit-rejected path: the first bundle was
        stored and then refused, and the failure bundle that answers for
        it is a second effect for one attempt. The journal records one
        phase per attempt, so the second bundle is stored — it is
        content-addressed and cheap — and the phase is left naming the
        first, which is the one the run actually tried to commit.
        """
        if self.bundles is None:
            return
        store_facts(bundle, self.store)
        bundle_id = self.bundles.record(bundle)
        if self.journal is None or replacing:
            return
        self.journal.record(
            attempt_id=attempt_id,
            phase=AttemptPhase.BUNDLE_DURABLE,
            bundle_id=bundle_id,
            detail=f"{bundle.outcome.status}",
        )

    def _committed(self, attempt_id: str, successor: ResearchState) -> None:
        """The successor exists; make it durable and say so.

        Persisted here rather than only at the end of the step, because
        the phase claims a state a recovering process must be able to
        load. A durability claim written before the bytes exist is the
        one thing this whole record must never do.
        """
        if self.journal is None:
            return
        # The successor's ancestors go down with it. A snapshot whose
        # parents are not stored is exactly the broken lineage the
        # verifier refuses to call intact, and the rest of the step —
        # which may still fail — is what would otherwise have written
        # them.
        for produced in recorded_lineage() or (successor,):
            self._persist(produced)
        self.journal.record(
            attempt_id=attempt_id,
            phase=AttemptPhase.COMMITTED,
            state_id=successor.id,
        )

    def _bill(
        self,
        state: ResearchState,
        actual: ResourceCost,
        estimated: ResourceCost,
        *,
        charge_id: str,
    ) -> tuple[ResearchState, str | None, bool]:
        """Charge the state, settle the hold, and close the attempt.

        Two details decide correctness. The ledger receives exactly what
        came off the state's budget, derived from the two balances rather
        than taken from the caller, so the two records cannot drift
        apart. And the balance is checked against the state afterwards,
        so a divergence raises here instead of travelling on as a halt
        reason — a bookkeeping failure is not a research outcome.
        """
        before = state.budget
        state, note, exhausted = _reconcile_cost(state, actual, estimated)
        charged = _charged_between(before, state.budget)
        self._settle(charged, charge_id=charge_id, estimated=estimated)
        if self.ledger is not None:
            self.ledger.require_balance(state.budget)
        if self.journal is not None and not estimated.is_zero:
            self.journal.record(
                attempt_id=charge_id,
                phase=AttemptPhase.COMPLETED,
                reserved=estimated,
                settled=charged,
                # A live step knows what came off the budget, because it
                # is the one that took it off.
                basis=SettlementBasis.MEASURED,
            )
        return state, note, exhausted

    def _settle(
        self,
        charged: ResourceCost,
        *,
        charge_id: str,
        estimated: ResourceCost,
    ) -> None:
        """Answer this attempt's hold with what came off the budget.

        Every debit answers a reservation — no exceptions, or a verifier
        could not check the link. Where no hold was taken in advance
        (nothing was expected to be spent, and something was) one is
        posted now for the larger of the two figures, with a reason
        saying it was authorized late. That is worse authorization than
        holding the money first, and it is still a complete record.
        """
        if self.ledger is None:
            return
        held = self.ledger.holds(charge_id)
        if charged.is_zero and not held:
            return  # nothing held and nothing spent
        if not held:
            self.ledger.reserve(
                charged if charged.exceeds(estimated) else estimated,
                charge_id=charge_id,
                reason=f"attempt {charge_id} (authorized at settlement)",
            )
        self.ledger.settle(
            charged, charge_id=charge_id, reason=f"attempt {charge_id}"
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
    """Bill the work that just committed; never leave it unbilled, and
    never bill less than it cost.

    Returns ``(state, note, exhausted)``. The actual cost is charged in
    full whatever it is. An overrun beyond the invocation's estimate is
    noted; one beyond the remaining budget is noted, takes the balance
    below zero, and halts the run.

    Charging the full figure past an empty budget is the point. The
    earlier version clamped — it charged the largest affordable share and
    posted that to the ledger — which kept the two records agreeing by
    making both of them wrong. Money spent above the budget then appeared
    nowhere at all, which is precisely the failure a budget exists to
    make visible. A negative remainder is unpleasant to read and it is
    true, and the run stops either way.
    """
    if state.budget.can_afford(actual):
        note = None
        if actual.exceeds(estimated):
            note = (
                "budget overrun: actual cost exceeded the invocation's "
                "estimated budget; charged in full"
            )
        return state.charge(actual), note, False
    note = (
        "budget breach: actual cost exceeded the remaining budget; "
        "charged in full, the balance is negative, and the program "
        "halted"
    )
    return state.charge(actual, allow_overdraw=True), note, True


def _charged_between(
    before: ResearchBudget, after: ResearchBudget
) -> ResourceCost:
    """What actually came off the budget. Derived from the two balances
    rather than taken from the caller, so the ledger and the state cannot
    disagree about the figure even by a rounding error."""
    return ResourceCost(
        wall_clock_seconds=before.wall_clock_seconds - after.wall_clock_seconds,
        gpu_hours=before.gpu_hours - after.gpu_hours,
        usd=before.usd - after.usd,
        model_tokens=before.model_tokens - after.model_tokens,
    )




# -- helpers -----------------------------------------------------------------


def _implementation_failures(
    report: VerificationReport,
) -> tuple[VerificationCheck, ...]:
    """The implementation-dimension FAIL checks of one report — the only
    admissible evidence for implementation repair."""
    return tuple(
        check
        for check in report.checks
        if check.dimension is ValidityDimension.IMPLEMENTATION
        and check.state is CheckState.FAIL
    )


def _assessed_hypothesis(state: ResearchState, subject_id: str) -> str | None:
    """The hypothesis a new assessment bears on — directly, or through the
    claim it judges. ``None`` when the subject reaches no hypothesis."""
    if state.hypothesis(subject_id) is not None:
        return subject_id
    claim = state.claim(subject_id)
    return claim.hypothesis_id if claim is not None else None


def _conclusive_evidence_ids(
    state: ResearchState,
    store: EvidenceStore,
    hypothesis_id: str,
    *,
    admissible: ScientificAdmissibility | None = None,
) -> tuple[str, ...]:
    """Evidence resting on conclusive tests of the hypothesis's predictions.

    With ``admissible`` given, only scientifically admissible results
    qualify — what a complete *trusted* analysis must cover. Without it,
    the full mechanical family — what an assessor should still be able to
    *see* (context visibility, not coverage obligation).
    """
    conclusive_results = {
        test.result_id
        for prediction in state.predictions_for(hypothesis_id)
        for test in state.tests_for(prediction.id)
        if test.consistency is not Consistency.INCONCLUSIVE
        and (admissible is None or admissible(test.result_id))
    }
    return tuple(
        evidence_id
        for evidence_id in state.evidence_ids
        if store.get_evidence(evidence_id).result_id in conclusive_results
    )


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


def _terminal_failure_reason(error: Exception) -> str | None:
    """A reason to halt the run after this failure's durable record, or
    ``None`` when a later dispatch could plausibly end differently.

    A rejected credential fails identically on every retry until a human
    rotates the key (observed live as ten consecutive billed 401s), and a
    local configuration mistake repeats until a human fixes it — so both
    end the run after exactly one provider call and one durable failed
    attempt, with no repair call and no scientific-state mutation. A
    preflight failure is terminal only when a check that indicts the host
    environment failed; a defect in the job under check is the generating
    role's to fix on a later attempt and stays retryable.
    """
    if isinstance(error, ProviderAuthenticationError):
        return (
            f"terminal provider failure: {error} — the credential is "
            f"rejected, and no retry within this run can change that"
        )
    if isinstance(error, ProviderConfigurationError):
        return (
            f"terminal configuration failure: {error} — a human must fix "
            f"the provider configuration before any retry can differ"
        )
    if isinstance(error, PreflightError):
        stable = sorted(
            check.name
            for check in error.report.failures
            if check.name in TERMINAL_ENVIRONMENT_CHECKS
        )
        if stable:
            return (
                f"terminal environment failure: {', '.join(stable)} — the "
                f"host environment, not the job under check, is broken; "
                f"re-dispatching identical work would re-diagnose the "
                f"same host"
            )
    return None


def _outcome_of(state: ResearchState, attempt_id: str) -> ActionOutcome:
    attempt = next(a for a in state.attempts if a.id == attempt_id)
    assert attempt.outcome is not None
    return attempt.outcome


def _standing_notes(
    evidence: tuple[Evidence, ...], verifications: VerificationStore
) -> tuple[str, ...]:
    """Verification standing, projected into a role's context notes.

    This is how a scientific reasoning seat distinguishes verified evidence
    from observed-but-unresolved evidence without receiving global state:
    each piece of evidence it is shown arrives annotated with the durable
    verdict governing its result. Evidence without a record carries no
    annotation — it predates or was excluded from verification.
    """
    notes: list[str] = []
    for item in evidence:
        verdict = verifications.get(item.result_id)
        if verdict is None:
            continue
        notes.append(
            f"verification: evidence {item.id} (result {item.result_id}) "
            f"stands at {verdict.validity} — {verdict.standing}"
        )
    return tuple(notes)


def _context_for(
    state: ResearchState,
    store: EvidenceStore,
    verifications: VerificationStore,
    action: ResearchAction,
    admissible: ScientificAdmissibility,
) -> RoleContext:
    """The projection each seat receives: exactly what the assignment needs.

    This is where worker lifetime is separated from lab lifetime — an
    executor sees a spec and its prior runs, never the research history.
    Evidence shown to scientific seats arrives annotated with its durable
    verification standing (:func:`_standing_notes`).
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
            return _synthesis_context(state, store, verifications, target)
        case ResearchActionType.ASSESS_CLAIM:
            return _assessment_context(state, store, verifications, target)
        case ResearchActionType.PLAN_NEXT_ACTION:
            return _planning_context(state, store, verifications, admissible)
        case _:
            return RoleContext(objective=state.objective)


#: The explicit bound on how many of the most recent results the planning
#: projection carries — the prompt stays finite as history grows.
_MAX_PLANNING_RESULTS = 32


def _planning_context(
    state: ResearchState,
    store: EvidenceStore,
    verifications: VerificationStore,
    admissible: ScientificAdmissibility,
) -> RoleContext:
    """The planner's deterministic, bounded projection: the whole
    scientific chain from authoritative state, every piece of evidence
    annotated with admissibility, standing notes and contradictions in the
    notes, and the remaining budget — never conversation history."""
    results = tuple(
        store.get_result(ref.result_id)
        for ref in state.results[-_MAX_PLANNING_RESULTS:]
    )
    evidence = tuple(
        store.get_evidence(evidence_id) for evidence_id in state.evidence_ids
    )
    contradiction_notes = tuple(
        f"contradiction: {c.subject_kind} {c.subject_id} — {c.detail}"
        for c in find_contradictions(state, admissible=admissible)
    )
    return RoleContext(
        objective=state.objective,
        questions=state.questions,
        hypotheses=state.hypotheses,
        predictions=state.predictions,
        prediction_tests=state.prediction_tests,
        experiments=state.experiments,
        results=results,
        evidence=evidence,
        claims=state.claims,
        evidence_links=state.evidence_links,
        assessments=state.assessments,
        notes=(*_standing_notes(evidence, verifications), *contradiction_notes),
        admissible_evidence_ids=tuple(
            item.id for item in evidence if admissible(item.result_id)
        ),
        remaining_budget=state.budget,
    )


def _synthesis_context(
    state: ResearchState,
    store: EvidenceStore,
    verifications: VerificationStore,
    evidence_id: str | None,
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
        notes=_standing_notes((evidence,), verifications),
    )


def _assessment_context(
    state: ResearchState,
    store: EvidenceStore,
    verifications: VerificationStore,
    claim_id: str | None,
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
    # An assessor sees — and can therefore cite — every conclusive
    # observation bearing on the hypothesis, not only the claim's own
    # links: the analysis-validity gate holds it to exactly that coverage.
    linked_ids = tuple(link.evidence_id for link in links)
    family_ids = (
        _conclusive_evidence_ids(state, store, hypothesis.id)
        if hypothesis is not None
        else ()
    )
    evidence_ids = tuple(dict.fromkeys((*linked_ids, *family_ids)))
    evidence = tuple(store.get_evidence(eid) for eid in evidence_ids)
    return RoleContext(
        objective=state.objective,
        hypotheses=(hypothesis,) if hypothesis else (),
        predictions=tuple(predictions),
        prediction_tests=tests,
        evidence=evidence,
        claims=(claim,),
        evidence_links=links,
        assessments=state.assessments,
        notes=_standing_notes(evidence, verifications),
    )


def _critic_context(
    state: ResearchState,
    store: EvidenceStore,
    verifications: VerificationStore,
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
        notes=(*reasons, *_standing_notes(evidence, verifications)),
    )
