"""What one attempt did, phase by phase, written down as it happened.

Layout, under a program root::

    <root>/
    └── journals/
        └── <run_id>/
            ├── 000000.json      att_1, started
            ├── 000001.json      att_1, submitted
            └── ...

The mechanism is the budget ledger's and the stage log's, for the same
reasons: sequence numbers as filenames so a gap is visible without
reading anything, each event naming the id of the event before it so a
deletion or a reordering contradicts the chain, and publication by
hard-linking a scratch file into place so a crash mid-write leaves an
ignorable temporary rather than a corrupt journal.

What it is *for* is the gap the ledger cannot close on its own. A ledger
says money moved; it cannot say what the money bought, or whether the
thing it bought was ever written down. A process killed between paying
and recording leaves those two records disagreeing, and nothing to
decide between them. This is that deciding record: for every attempt,
how far it got, which job it used, which bundle it produced, and which
state it committed.

Read together with the ledger it answers the only question recovery
actually has — *what does this run still owe, and what has it already
paid for?* — with a fact rather than an inference:

============================  =============================================
last phase                    what is true
============================  =============================================
``STARTED``                   nothing has run *if* no job exists under
                              the derived id; release and retry
``SUBMITTED``                 a job may exist under a known id; collect
                              it, never resubmit it
``OUTPUTS_DURABLE``           the work is bought and collectable
``BUNDLE_DURABLE``            the successor is derivable without the
                              runtime; apply the bundle
``COMMITTED``                 the successor is stored; only the money is
                              still open
terminal                      nothing is owed
============================  =============================================

A budget breach is not a phase. It is the closing event whose measured
``settled`` exceeds its ``reserved``, which is two numbers already on
the record rather than a third field that could disagree with them.

*Measured* is load-bearing there. An attempt a crash left unaccounted
for is charged its authorization, because that is the only figure the
run can defend — but it did not *cost* that, and the closing event says
so by carrying ``CONSERVATIVE_MAX`` rather than a basis it has no right
to claim. The ledger stays safe either way; this is what keeps it
truthful as well.

One writer per run is assumed, as everywhere else in the repository. The
exclusive create still makes a collision loud rather than silent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.attempt import AttemptPhase, SettlementBasis
from ..core.budget import NO_COST, ResourceCost
from ..core.ids import content_id, occurrence_id

_JOURNALS: Final = "journals"
_EVENT_SUFFIX: Final = ".json"
_SEQUENCE_DIGITS: Final = 6
_MAX_APPEND_ATTEMPTS: Final = 16
"""How many times a losing writer re-reads the head and tries again.
Bounded so a pathological contender fails loudly instead of spinning."""

MAX_DETAIL_CHARS: Final = 400
"""A detail says what happened in one breath — an error's message, a
recovery's reason. The full account lives in the records themselves."""

_ORDER: Final[Mapping[AttemptPhase, int]] = {
    AttemptPhase.STARTED: 0,
    AttemptPhase.SUBMITTED: 1,
    AttemptPhase.OUTPUTS_DURABLE: 2,
    AttemptPhase.BUNDLE_DURABLE: 3,
    AttemptPhase.COMMITTED: 4,
    AttemptPhase.COMPLETED: 5,
    AttemptPhase.RELEASED: 5,
    AttemptPhase.ABANDONED: 5,
}
"""How far along each phase is. Phases may be skipped but never
repeated and never reversed, so one comparison enforces the lifecycle."""

_SETTLING: Final = frozenset(
    {AttemptPhase.COMPLETED, AttemptPhase.ABANDONED}
)
"""The phases that close an attempt by answering its reservation, and so
the only ones that may say what it cost."""


class JournalIntegrityError(RuntimeError):
    """The stored journal contradicts itself: a gap, a broken chain, an
    event filed under an id it no longer derives, or unreadable JSON."""


class JournalConflictError(RuntimeError):
    """One attempt's phase was recorded twice with different content, or
    a phase would go backwards."""


class JournalContentionError(RuntimeError):
    """An append lost its race for a sequence number too many times."""


@dataclass(frozen=True, slots=True)
class AttemptEvent:
    """One attempt reaching one phase."""

    run_id: str
    sequence: int
    attempt_id: str
    phase: AttemptPhase

    state_id: str = ""
    """The state this event is about: the one the attempt begins from on
    ``STARTED``, the successor it produced on ``COMMITTED``. One field
    because it is one question — which snapshot does this phase name —
    and no phase ever needs to answer it twice."""

    job_id: str = ""
    """The executor job, preallocated before submission so that recovery
    recomputes it rather than looking it up. Empty for attempts that run
    no job."""

    bundle_id: str = ""
    produced: tuple[tuple[str, str], ...] = ()
    """The ids this phase made durable, as sorted name/id pairs — the
    result and evidence on ``OUTPUTS_DURABLE``, whatever else a phase
    brought into being."""

    reserved: ResourceCost = NO_COST
    """The authorized maximum held for this attempt. Recorded on
    ``STARTED``, before the reservation is posted."""

    settled: ResourceCost = NO_COST
    """What the attempt was charged, recorded on the event that closes
    it. Kept apart from ``reserved`` because the gap between them is the
    whole point: an attempt that overran is recorded as having
    overrun."""

    basis: SettlementBasis = SettlementBasis.NONE
    """Where ``settled`` came from — a measurement, or the authorized
    maximum charged because nobody knows what the attempt cost. Without
    this the two are indistinguishable on the record, and a deliberate
    over-charge would read as a figure someone took."""

    detail: str = ""
    previous_event_id: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("attempt_id", self.attempt_id),
        ):
            if not value.strip():
                raise ValueError(f"an attempt event must name its {name}")
        if self.sequence < 0:
            raise ValueError(
                f"an event sequence starts at zero, got {self.sequence}"
            )
        if len(self.detail) > MAX_DETAIL_CHARS:
            raise ValueError(
                f"detail must be at most {MAX_DETAIL_CHARS} characters, "
                f"got {len(self.detail)}"
            )
        self._require_what_the_phase_promises()
        names = [name for name, _ in self.produced]
        if len(names) != len(set(names)):
            raise ValueError("one event produces one id per name")
        if list(self.produced) != sorted(self.produced):
            raise ValueError(
                "produced ids are stored sorted, so one event has one "
                "serialization and one id"
            )
        for name, value in self.produced:
            if not name.strip() or not value.strip():
                raise ValueError("a produced id needs both a name and an id")
        if bool(self.previous_event_id) == (self.sequence == 0):
            raise ValueError(
                "every event but the first names its predecessor, and "
                "the first names none"
            )
        if not self.id:
            object.__setattr__(self, "id", self._derived_id())

    def _require_what_the_phase_promises(self) -> None:
        """Each phase claims something specific; an event that does not
        carry it is a phase nobody could act on."""
        if self.phase is AttemptPhase.STARTED:
            if not self.state_id.strip():
                raise ValueError(
                    "a started attempt names the state it begins from; "
                    "without it recovery has nothing to re-run against"
                )
            if self.reserved.is_zero:
                raise ValueError(
                    "a started attempt names the maximum it is authorized "
                    "to spend; an unreserved attempt is money nobody is "
                    "holding"
                )
        if self.phase is AttemptPhase.SUBMITTED and not self.job_id.strip():
            raise ValueError(
                "a submitted attempt names its job; recovery reattaches "
                "to that id or proves it was never submitted"
            )
        if (
            self.phase is AttemptPhase.BUNDLE_DURABLE
            and not self.bundle_id.strip()
        ):
            raise ValueError(
                "a durable bundle has an id; that id is what makes "
                "re-applying it idempotent"
            )
        if self.phase is AttemptPhase.COMMITTED and not self.state_id.strip():
            raise ValueError(
                "a committed attempt names the successor it produced"
            )
        if self.phase not in _SETTLING and not self.settled.is_zero:
            # ``RELEASED`` is caught here too, and that is the point: an
            # attempt that bought nothing cannot report a cost.
            raise ValueError(
                "only the event that closes an attempt by settling it says "
                "what it cost; a cost recorded anywhere else is a second "
                "answer"
            )
        if (self.phase in _SETTLING) != (
            self.basis is not SettlementBasis.NONE
        ):
            raise ValueError(
                f"a settling event states where its figure came from and "
                f"nothing else may; {self.phase} carries {self.basis}"
            )

    def _derived_id(self) -> str:
        return content_id(
            "aevt",
            self.run_id,
            self.sequence,
            self.attempt_id,
            str(self.phase),
            self.state_id,
            self.job_id,
            self.bundle_id,
            self.produced,
            self.reserved.wall_clock_seconds,
            self.reserved.gpu_hours,
            self.reserved.usd,
            self.reserved.model_tokens,
            self.settled.wall_clock_seconds,
            self.settled.gpu_hours,
            self.settled.usd,
            self.settled.model_tokens,
            str(self.basis),
            self.detail,
            self.previous_event_id,
        )

    @property
    def actual_cost_known(self) -> bool:
        """Whether ``settled`` is what the attempt cost, or only what it
        was charged."""
        return self.basis is SettlementBasis.MEASURED

    @property
    def breached(self) -> bool:
        """Whether this event records an attempt that cost more than it
        was authorized to.

        Only a measurement can breach. A conservative charge is the
        authorization by construction, and calling that an overrun would
        turn every crash into a budget incident."""
        return self.actual_cost_known and self.settled.exceeds(self.reserved)


class RunJournal:
    """The attempt journal of one run, addressed by a program root and a
    run id."""

    def __init__(self, root: Path | str, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("a journal belongs to a named run")
        self._run_id = run_id
        self._directory = Path(root) / _JOURNALS / run_id
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def directory(self) -> Path:
        return self._directory

    # -- reading ---------------------------------------------------------------

    def events(self) -> tuple[AttemptEvent, ...]:
        """Every event in sequence order, verified on the way out: ids
        re-derived, positions checked, the chain walked."""
        paths = sorted(self._directory.glob(f"*{_EVENT_SUFFIX}"))
        events: list[AttemptEvent] = []
        previous_id = ""
        for position, path in enumerate(paths):
            event = self._read(path)
            if event.sequence != position:
                raise JournalIntegrityError(
                    f"journal {self._run_id} jumps from sequence "
                    f"{position - 1} to {event.sequence}; an event is "
                    f"missing or misnamed"
                )
            if event.run_id != self._run_id:
                raise JournalIntegrityError(
                    f"event {path.name} belongs to run {event.run_id}, "
                    f"not {self._run_id}"
                )
            if event.previous_event_id != previous_id:
                raise JournalIntegrityError(
                    f"event {path.name} follows "
                    f"{event.previous_event_id or 'nothing'}, but the "
                    f"event before it is {previous_id or 'nothing'}"
                )
            events.append(event)
            previous_id = event.id
        return tuple(events)

    def attempts(self) -> tuple[str, ...]:
        """Every attempt id the journal knows, in the order each began."""
        seen: list[str] = []
        for event in self.events():
            if event.attempt_id not in seen:
                seen.append(event.attempt_id)
        return tuple(seen)

    def events_for(self, attempt_id: str) -> tuple[AttemptEvent, ...]:
        return tuple(
            event
            for event in self.events()
            if event.attempt_id == attempt_id
        )

    def last_for(self, attempt_id: str) -> AttemptEvent | None:
        """How far one attempt got. ``None`` means the journal has never
        heard of it."""
        events = self.events_for(attempt_id)
        return events[-1] if events else None

    def event_at(
        self, attempt_id: str, phase: AttemptPhase
    ) -> AttemptEvent | None:
        return next(
            (e for e in self.events_for(attempt_id) if e.phase is phase),
            None,
        )

    def open_attempts(self) -> tuple[str, ...]:
        """Attempts that reached no terminal phase — what a crash leaves
        behind, and the whole input to recovery."""
        return tuple(
            attempt_id
            for attempt_id in self.attempts()
            if not _reached_terminal(self.events_for(attempt_id))
        )

    def breaches(self) -> tuple[AttemptEvent, ...]:
        """Every attempt that cost more than it was authorized to."""
        return tuple(event for event in self.events() if event.breached)

    # -- writing ---------------------------------------------------------------

    def record(
        self,
        *,
        attempt_id: str,
        phase: AttemptPhase,
        state_id: str = "",
        job_id: str = "",
        bundle_id: str = "",
        produced: Iterable[tuple[str, str]] = (),
        reserved: ResourceCost = NO_COST,
        settled: ResourceCost = NO_COST,
        basis: SettlementBasis = SettlementBasis.NONE,
        detail: str = "",
    ) -> AttemptEvent:
        """Publish one phase at the end of the journal.

        Recording the same phase for the same attempt twice records it
        once and returns what is already written, because recovery
        re-drives phases it cannot prove happened. The same phase
        claiming a different job, bundle, state, or cost is a conflict:
        an attempt has one history, and two versions of it means the
        deterministic derivation stopped being deterministic. ``detail``
        is excluded from that comparison — a recovered phase may
        reasonably explain itself differently from the first attempt at
        it.
        """
        pairs = tuple(sorted(produced))
        for _ in range(_MAX_APPEND_ATTEMPTS):
            events = self.events()
            mine = tuple(e for e in events if e.attempt_id == attempt_id)
            existing = next((e for e in mine if e.phase is phase), None)
            if existing is not None:
                return _require_same_phase(
                    existing,
                    state_id=state_id,
                    job_id=job_id,
                    bundle_id=bundle_id,
                    produced=pairs,
                    reserved=reserved,
                    settled=settled,
                    basis=basis,
                )
            _require_forward(mine, phase, attempt_id)
            event = AttemptEvent(
                run_id=self._run_id,
                sequence=len(events),
                attempt_id=attempt_id,
                phase=phase,
                state_id=state_id,
                job_id=job_id,
                bundle_id=bundle_id,
                produced=pairs,
                reserved=reserved,
                settled=settled,
                basis=basis,
                detail=detail,
                previous_event_id=events[-1].id if events else "",
            )
            if self._publish(event):
                return event
            # Another writer took this sequence number; re-read the head
            # and rebuild against it.
        raise JournalContentionError(
            f"journal {self._run_id} lost {_MAX_APPEND_ATTEMPTS} races for a "
            f"sequence number; refusing to keep trying"
        )

    def _publish(self, event: AttemptEvent) -> bool:
        """Hard-link the event into place. ``False`` means the sequence
        was taken while this event was being built."""
        target = self._path(event.sequence)
        scratch = self._directory / f"{occurrence_id('jscratch')}.tmp"
        scratch.write_text(
            json.dumps(_event_payload(event), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            os.link(scratch, target)
        except FileExistsError:
            return False
        finally:
            scratch.unlink(missing_ok=True)
        return True

    # -- files -----------------------------------------------------------------

    def _path(self, sequence: int) -> Path:
        return (
            self._directory / f"{sequence:0{_SEQUENCE_DIGITS}d}{_EVENT_SUFFIX}"
        )

    def _read(self, path: Path) -> AttemptEvent:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise JournalIntegrityError(
                f"attempt event {path.name} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise JournalIntegrityError(
                f"attempt event {path.name} is not an object"
            )
        try:
            event = _event_from(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise JournalIntegrityError(
                f"attempt event {path.name} cannot be read: {exc}"
            ) from exc
        filed_as = payload.get("id")
        if filed_as != event.id:
            raise JournalIntegrityError(
                f"attempt event {path.name} claims id {filed_as!r} but "
                f"re-derives {event.id}; the file was edited"
            )
        return event


_ENDS_EARLY: Final = frozenset(
    {AttemptPhase.RELEASED, AttemptPhase.ABANDONED}
)
"""The two ways an attempt ends without a successor."""


def _reached_terminal(events: tuple[AttemptEvent, ...]) -> bool:
    return any(event.phase.is_terminal for event in events)


def _require_forward(
    mine: tuple[AttemptEvent, ...], phase: AttemptPhase, attempt_id: str
) -> None:
    """An attempt's history only ever grows forwards."""
    if not mine:
        if phase is not AttemptPhase.STARTED:
            raise JournalConflictError(
                f"attempt {attempt_id} has no history, so it cannot "
                f"already be {phase}; every attempt begins at "
                f"{AttemptPhase.STARTED}"
            )
        return
    last = mine[-1]
    if _ORDER[phase] <= _ORDER[last.phase]:
        raise JournalConflictError(
            f"attempt {attempt_id} is already {last.phase}; recording "
            f"{phase} would move it backwards, and an attempt has one "
            f"history"
        )
    if phase in _ENDS_EARLY and any(
        e.phase is AttemptPhase.COMMITTED for e in mine
    ):
        raise JournalConflictError(
            f"attempt {attempt_id} already committed a successor; it "
            f"cannot then be {phase}, which says no state change came of "
            f"it"
        )


def _require_same_phase(
    existing: AttemptEvent,
    *,
    state_id: str,
    job_id: str,
    bundle_id: str,
    produced: tuple[tuple[str, str], ...],
    reserved: ResourceCost,
    settled: ResourceCost,
    basis: SettlementBasis,
) -> AttemptEvent:
    """One phase per attempt, recording one set of facts."""
    same = (
        existing.state_id == state_id
        and existing.job_id == job_id
        and existing.bundle_id == bundle_id
        and existing.produced == produced
        and existing.reserved == reserved
        and existing.settled == settled
        and existing.basis is basis
    )
    if not same:
        raise JournalConflictError(
            f"attempt {existing.attempt_id} already recorded "
            f"{existing.phase} naming state {existing.state_id or 'none'}, "
            f"job {existing.job_id or 'none'}, bundle "
            f"{existing.bundle_id or 'none'}; refusing to record a second "
            f"version of one phase"
        )
    return existing


def _event_payload(event: AttemptEvent) -> dict[str, object]:
    return {
        "id": event.id,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "attempt_id": event.attempt_id,
        "phase": str(event.phase),
        "state_id": event.state_id,
        "job_id": event.job_id,
        "bundle_id": event.bundle_id,
        "produced": dict(event.produced),
        "reserved": _amounts(event.reserved),
        "settled": _amounts(event.settled),
        "basis": str(event.basis),
        "detail": event.detail,
        "previous_event_id": event.previous_event_id,
    }


def _event_from(payload: Mapping[str, object]) -> AttemptEvent:
    produced = payload["produced"]
    if not isinstance(produced, dict):
        raise TypeError("produced must be an object of name/id pairs")
    return AttemptEvent(
        run_id=_text(payload, "run_id"),
        sequence=_integer(payload, "sequence"),
        attempt_id=_text(payload, "attempt_id"),
        phase=AttemptPhase(_text(payload, "phase")),
        state_id=_text(payload, "state_id"),
        job_id=_text(payload, "job_id"),
        bundle_id=_text(payload, "bundle_id"),
        produced=tuple(sorted(_pairs(produced))),
        reserved=_cost(payload, "reserved"),
        settled=_cost(payload, "settled"),
        basis=SettlementBasis(_text(payload, "basis")),
        detail=_text(payload, "detail"),
        previous_event_id=_text(payload, "previous_event_id"),
    )


def _amounts(cost: ResourceCost) -> dict[str, float | int]:
    return {
        "wall_clock_seconds": cost.wall_clock_seconds,
        "gpu_hours": cost.gpu_hours,
        "usd": cost.usd,
        "model_tokens": cost.model_tokens,
    }


def _cost(payload: Mapping[str, object], key: str) -> ResourceCost:
    raw = payload[key]
    if not isinstance(raw, dict):
        raise TypeError(f"{key} must be an object of resource dimensions")
    return ResourceCost(
        wall_clock_seconds=_number(raw, "wall_clock_seconds"),
        gpu_hours=_number(raw, "gpu_hours"),
        usd=_number(raw, "usd"),
        model_tokens=_integer(raw, "model_tokens"),
    )


def _pairs(produced: Mapping[str, object]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, value in produced.items():
        if not isinstance(value, str):
            raise TypeError(f"produced id {name} must be a string")
        pairs.append((name, value))
    return pairs


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)
