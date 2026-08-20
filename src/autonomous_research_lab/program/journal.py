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
``STARTED``                   nothing has run; the reservation may be
                              released and the attempt abandoned
``SUBMITTED``                 a job may exist under a known id; reattach
                              or collect, never resubmit
``OUTPUTS_DURABLE``           the work is bought; finish committing it
``BUNDLE_DURABLE``            the successor is derivable without the
                              runtime; apply the bundle
``COMMITTED``                 the money is settled and the state is
                              stored; only the closing record is missing
``COMPLETED`` / ``RELEASED``  nothing is owed
============================  =============================================

A budget breach is not a phase. It is the ``COMMITTED`` event whose
``actual`` exceeds its ``reserved``, which is two numbers already on the
record rather than a third field that could disagree with them.

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

from ..core.attempt import AttemptPhase
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
}
"""How far along each phase is. Phases may be skipped but never
repeated and never reversed, so one comparison enforces the lifecycle."""


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

    actual: ResourceCost = NO_COST
    """What the attempt really cost, recorded on ``COMMITTED``. Kept
    apart from ``reserved`` because the gap between them is the whole
    point: an attempt that overran is recorded as having overrun."""

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
        if self.phase is not AttemptPhase.COMMITTED and not self.actual.is_zero:
            raise ValueError(
                "only the committing event says what an attempt cost; a "
                "cost recorded anywhere else is a second answer"
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
            self.actual.wall_clock_seconds,
            self.actual.gpu_hours,
            self.actual.usd,
            self.actual.model_tokens,
            self.detail,
            self.previous_event_id,
        )

    @property
    def breached(self) -> bool:
        """Whether this event records an attempt that cost more than it
        was authorized to."""
        return (
            self.phase is AttemptPhase.COMMITTED
            and self.actual.exceeds(self.reserved)
        )


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
        actual: ResourceCost = NO_COST,
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
                    actual=actual,
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
                actual=actual,
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
    if (
        phase is AttemptPhase.RELEASED
        and any(e.phase is AttemptPhase.COMMITTED for e in mine)
    ):
        raise JournalConflictError(
            f"attempt {attempt_id} settled its debit; releasing it now "
            f"would give back money that has already been spent"
        )


def _require_same_phase(
    existing: AttemptEvent,
    *,
    state_id: str,
    job_id: str,
    bundle_id: str,
    produced: tuple[tuple[str, str], ...],
    reserved: ResourceCost,
    actual: ResourceCost,
) -> AttemptEvent:
    """One phase per attempt, recording one set of facts."""
    same = (
        existing.state_id == state_id
        and existing.job_id == job_id
        and existing.bundle_id == bundle_id
        and existing.produced == produced
        and existing.reserved == reserved
        and existing.actual == actual
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
        "actual": _amounts(event.actual),
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
        actual=_cost(payload, "actual"),
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
