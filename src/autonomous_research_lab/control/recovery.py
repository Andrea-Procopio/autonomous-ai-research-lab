"""Finishing what a killed process started.

Task 6C made a run resumable at stage boundaries. Inside a step it was
not: a process that died between paying for an experiment and recording
what the money bought left the ledger and the snapshots disagreeing, and
the next step correctly refused to guess which was true. This is what
answers that question instead of guessing — the attempt journal, read
before the chain steps again.

Every open attempt gets one of two answers, and which one depends on a
single fact: whether its commit bundle reached disk.

**The bundle is durable.** Then the whole effect of the attempt survives
— the proposals, the outcome, and what it cost — and applying it again
reaches the same successor with the same id. Recovery applies it, charges
the state, settles the ledger with the cost the bundle records, and
closes the attempt as ``COMPLETED``. Nothing is lost and nothing is
re-run. This is the expensive case, and it is the one that recovers
completely.

**The bundle is not durable.** Then nothing on disk says what the attempt
cost, and something almost certainly was spent: a model call, a job, or
both. Recovery settles the reservation *in full* and closes the attempt
as ``ABANDONED``. That can overcharge — an attempt killed a millisecond
after it began pays its whole authorization — and that is the deliberate
choice. The alternative is to release money that may well be gone, which
is the one thing this record exists to prevent. The authorized maximum is
the only number the run can defend, and erring toward recording more is
the direction to err in.

Two things recovery never does. It never resubmits a job: job ids are
derived from attempt ids, the executor refuses a second submission of
one, and a retry is a new attempt by definition. And it never deletes a
debit — a reservation already answered is left exactly as it is, which
is what makes running recovery twice a no-op.

It lives here, in the composition root, because it reads records from
``program`` and drives a transition from ``orchestration``, and those two
may not import each other. What it does not do is decide anything
scientific: the bundle it applies is the runtime's own decision, written
down before the crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..core.attempt import AttemptPhase
from ..core.budget import NO_COST, ResearchBudget, ResourceCost
from ..core.state import recording_lineage
from ..evidence.store import EvidenceStore
from ..orchestration.transitions import commit_bundle, reconcile_charge
from ..persistence.commit_store import CommitBundleStore
from ..persistence.state_store import FileStateStore
from ..program.journal import AttemptEvent, RunJournal
from ..program.ledger import BudgetLedger

_RECOVERABLE: Final = frozenset(
    {AttemptPhase.BUNDLE_DURABLE, AttemptPhase.COMMITTED}
)
"""The phases from which the whole attempt can be finished. Anything
earlier and there is no record of what it produced."""


@dataclass(frozen=True, slots=True)
class Recovery:
    """What recovery did to one interrupted attempt."""

    attempt_id: str
    left_at: AttemptPhase
    """How far the dead process got."""

    resolution: AttemptPhase
    """``COMPLETED`` when the attempt was finished, ``ABANDONED`` when it
    was paid for and closed without a successor."""

    state_id: str
    """Where the run continues from — the recovered successor, or the
    state the attempt began at."""

    settled: ResourceCost
    breached: bool
    detail: str

    @property
    def finished(self) -> bool:
        return self.resolution is AttemptPhase.COMPLETED


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Every open attempt, answered."""

    recoveries: tuple[Recovery, ...]
    state_id: str
    """Where the chain should continue from: the last recovered
    successor, the step that landed quietly, or the fallback."""

    adopted: str | None = None
    """A step that finished entirely and whose *stage event* was lost —
    a different crash from the ones the journal answers, and one it
    cannot pinpoint, because a step's last state is not an attempt
    boundary. Adopting it is what stops the chain paying for the same
    step twice."""

    settled: ResourceCost = NO_COST
    """What the state had to be charged to agree with the ledger again.

    The last thing recovery does, and the one thing it always does: a
    process can die between settling a debit and persisting the state
    that paid it, and the two records then differ by exactly this. It is
    the state that moves, never the ledger — the debit is what actually
    happened."""

    @property
    def anything_to_do(self) -> bool:
        return (
            bool(self.recoveries)
            or self.adopted is not None
            or not self.settled.is_zero
        )

    @property
    def breached(self) -> tuple[Recovery, ...]:
        return tuple(r for r in self.recoveries if r.breached)

    def summary(self) -> str:
        parts: list[str] = [r.detail for r in self.recoveries]
        if self.adopted is not None:
            parts.append(f"adopted the step a crash hid ({self.adopted})")
        if not self.settled.is_zero:
            parts.append(
                "charged the state the debit the ledger already carried"
            )
        return "; ".join(parts) if parts else "nothing was interrupted"


def recover(
    *,
    journal: RunJournal,
    ledger: BudgetLedger,
    bundles: CommitBundleStore,
    states: FileStateStore,
    evidence: EvidenceStore,
    fallback_state_id: str,
) -> RecoveryReport:
    """Answer every attempt the journal left open.

    ``fallback_state_id`` is where the run continues if nothing was
    interrupted, or if what was interrupted produced no successor.
    Running this twice does nothing the second time: the journal's
    terminal phases and the ledger's one-answer-per-hold rule both make
    the second pass find nothing to answer.
    """
    recoveries: list[Recovery] = []
    resume_from = fallback_state_id
    for attempt_id in journal.open_attempts():
        recovery = _answer(
            attempt_id,
            journal=journal,
            ledger=ledger,
            bundles=bundles,
            states=states,
            evidence=evidence,
            resume_from=resume_from,
        )
        recoveries.append(recovery)
        resume_from = recovery.state_id
    adopted: str | None = None
    if not recoveries:
        adopted = _the_step_that_landed_quietly(states, fallback_state_id)
        resume_from = adopted or fallback_state_id
    settled, resume_from = _reconciled(states, ledger, resume_from)
    return RecoveryReport(
        recoveries=tuple(recoveries),
        state_id=resume_from,
        adopted=adopted,
        settled=settled,
    )


def _answer(
    attempt_id: str,
    *,
    journal: RunJournal,
    ledger: BudgetLedger,
    bundles: CommitBundleStore,
    states: FileStateStore,
    evidence: EvidenceStore,
    resume_from: str,
) -> Recovery:
    events = journal.events_for(attempt_id)
    began = events[0]
    last = events[-1]
    if last.phase in _RECOVERABLE:
        return _finish(
            attempt_id,
            began=began,
            last=last,
            journal=journal,
            ledger=ledger,
            bundles=bundles,
            states=states,
            evidence=evidence,
        )
    return _abandon(
        attempt_id,
        began=began,
        last=last,
        journal=journal,
        ledger=ledger,
        resume_from=resume_from,
    )


def _finish(
    attempt_id: str,
    *,
    began: AttemptEvent,
    last: AttemptEvent,
    journal: RunJournal,
    ledger: BudgetLedger,
    bundles: CommitBundleStore,
    states: FileStateStore,
    evidence: EvidenceStore,
) -> Recovery:
    """Apply the stored bundle, or adopt the successor already applied."""
    durable = journal.event_at(attempt_id, AttemptPhase.BUNDLE_DURABLE)
    assert durable is not None  # the phase is what got us here
    bundle = bundles.load(durable.bundle_id, facts=evidence)
    cost = bundle.outcome.actual_cost
    committed = journal.event_at(attempt_id, AttemptPhase.COMMITTED)
    stored = set(states.state_ids())
    if committed is not None and committed.state_id in stored:
        successor_id = committed.state_id
    else:
        origin = states.load(began.state_id)
        # Every state the application derives goes down, not only the
        # last: a snapshot whose parents are missing is the broken
        # lineage the verifier refuses to call intact.
        with recording_lineage() as derived:
            successor = commit_bundle(origin, bundle, evidence)
        for produced in derived:
            states.persist(produced)
        successor_id = successor.id
        journal.record(
            attempt_id=attempt_id,
            phase=AttemptPhase.COMMITTED,
            state_id=successor_id,
            detail="applied the stored bundle after an interrupted step",
        )
    settlement = ledger.settle(
        cost, charge_id=attempt_id, reason=f"attempt {attempt_id}"
    )
    journal.record(
        attempt_id=attempt_id,
        phase=AttemptPhase.COMPLETED,
        reserved=began.reserved,
        actual=cost,
        detail=f"recovered from {last.phase}",
    )
    return Recovery(
        attempt_id=attempt_id,
        left_at=last.phase,
        resolution=AttemptPhase.COMPLETED,
        state_id=successor_id,
        settled=cost,
        breached=settlement.breached,
        detail=(
            f"{attempt_id} was interrupted at {last.phase}; its bundle was "
            f"applied and it settled {cost.usd} usd"
        ),
    )


def _abandon(
    attempt_id: str,
    *,
    began: AttemptEvent,
    last: AttemptEvent,
    journal: RunJournal,
    ledger: BudgetLedger,
    resume_from: str,
) -> Recovery:
    """Charge the authorization and close the attempt with nothing to
    show for it.

    Only the ledger moves here. The state is brought into agreement once,
    at the end, against the ledger's own balance — which is the same
    correction whether one attempt was abandoned or three, and which also
    catches the crash that landed between a settlement and the snapshot
    that recorded it.
    """
    if not ledger.holds(attempt_id):
        # The crash landed between writing the phase down and holding the
        # money. Nothing was authorized, so nothing could have been
        # spent through this attempt, and this is the one place a release
        # is provable rather than hopeful.
        journal.record(
            attempt_id=attempt_id,
            phase=AttemptPhase.RELEASED,
            detail="interrupted before anything was held for it",
        )
        return Recovery(
            attempt_id=attempt_id,
            left_at=last.phase,
            resolution=AttemptPhase.RELEASED,
            state_id=resume_from,
            settled=ResourceCost(),
            breached=False,
            detail=(
                f"{attempt_id} was interrupted at {last.phase} before "
                f"anything was held for it; nothing was spent"
            ),
        )
    cost = began.reserved
    settlement = ledger.settle(
        cost,
        charge_id=attempt_id,
        reason=f"attempt {attempt_id} (authorization charged in full)",
    )
    journal.record(
        attempt_id=attempt_id,
        phase=AttemptPhase.ABANDONED,
        reserved=began.reserved,
        actual=cost,
        detail=f"interrupted at {last.phase} with no durable bundle",
    )
    return Recovery(
        attempt_id=attempt_id,
        left_at=last.phase,
        resolution=AttemptPhase.ABANDONED,
        state_id=resume_from,
        settled=cost,
        breached=settlement.breached,
        detail=(
            f"{attempt_id} was interrupted at {last.phase} with no bundle "
            f"on disk; its authorization was charged in full and it "
            f"produced nothing"
        ),
    )


def _reconciled(
    states: FileStateStore, ledger: BudgetLedger, state_id: str
) -> tuple[ResourceCost, str]:
    """Bring the state's budget back to the ledger's balance.

    The last thing recovery does, and the one thing it always does. A
    process can die between settling a debit and persisting the state
    that paid it, so the two records differ by exactly the debit; the
    state is what moves, because the debit is what actually happened.

    The correction only ever goes one way. A state holding *less* than
    the ledger says would mean the ledger lost a movement, and inventing
    a credit to paper over that is how a bookkeeping failure becomes a
    bookkeeping fiction — it is left to the verifier to report.
    """
    state = states.load(state_id)
    gap = _gap(state.budget, ledger.balance())
    if gap.is_zero:
        return ResourceCost(), state_id
    with recording_lineage() as derived:
        charged = reconcile_charge(state, gap)
    for produced in derived:
        states.persist(produced)
    return gap, charged.id


def _gap(held: ResearchBudget, balance: ResearchBudget) -> ResourceCost:
    """How much the state is carrying that the ledger has already spent.
    Never negative in any dimension."""
    return ResourceCost(
        wall_clock_seconds=max(
            0.0, held.wall_clock_seconds - balance.wall_clock_seconds
        ),
        gpu_hours=max(0.0, held.gpu_hours - balance.gpu_hours),
        usd=max(0.0, held.usd - balance.usd),
        model_tokens=max(0, held.model_tokens - balance.model_tokens),
    )


def _the_step_that_landed_quietly(
    states: FileStateStore, state_id: str
) -> str | None:
    """The deepest descendant of ``state_id`` that is between steps.

    A different question from the journal's, and one the journal cannot
    answer. Its phases mark an *attempt's* boundaries, and a step is more
    than its attempt — verification, a critic, the charge — so the state
    a finished step leaves behind is not any phase's ``state_id``. What
    is true of it is structural: every attempt on it is resolved. A state
    with one still open is a step someone was in the middle of, and
    recovery has already dealt with those.
    """
    loaded = {found: states.load(found) for found in states.state_ids()}
    children: dict[str, list[str]] = {}
    for found, state in loaded.items():
        if state.parent_id:
            children.setdefault(state.parent_id, []).append(found)

    deepest: str | None = None
    depth = 0

    def walk(current: str, distance: int) -> None:
        nonlocal deepest, depth
        state = loaded.get(current)
        if (
            state is not None
            and distance > depth
            and all(attempt.status.is_terminal for attempt in state.attempts)
        ):
            deepest, depth = state.id, distance
        for child in children.get(current, ()):
            walk(child, distance + 1)

    walk(state_id, 0)
    return deepest
