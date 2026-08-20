"""The stage event log: what happened to one investigation, in order.

Layout, under a control root::

    <root>/
    └── logs/
        └── <investigation_id>/
            ├── 000000.json      mapping, running
            ├── 000001.json      mapping, succeeded
            └── ...

The mechanism is the budget ledger's, for the same reasons and with the
same properties: sequence numbers as filenames so a gap is visible
without reading anything, each event naming the id of the event before
it so a deletion or a reordering contradicts the chain, and publication
by hard-linking a scratch file into place so a crash mid-write leaves an
ignorable temporary rather than a corrupt log.

What differs is what the log is *for*. A ledger answers "what is left";
this answers "where did the process die, and what may be skipped". Two
events per stage carry that: ``RUNNING`` written before the side effect
and a terminal status written after it. A ``RUNNING`` with no terminal
successor is exactly the crash signature, and the controller reconciles
it by asking the stage's own store whether the work actually landed.

The log is also the resume mechanism. Every id a stage produces is
recorded on its terminal event, so a fresh process rebuilds the whole
chain's state by replaying the log — never by trusting an object that
outlived nothing.

One writer per investigation is assumed. Two controllers racing over one
root would each hold their own investigation and their own log, and the
stage stores underneath are write-once, so the collision is loud rather
than silent; it is still not a supported way to run the system.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.ids import content_id, occurrence_id
from .stage import (
    TERMINAL_STATUSES,
    ZERO_SPEND,
    ChainFacts,
    StageName,
    StageSpend,
    StageStatus,
)

_LOGS: Final = "logs"
_EVENT_SUFFIX: Final = ".json"
_SEQUENCE_DIGITS: Final = 6
_MAX_APPEND_ATTEMPTS: Final = 16
"""How many times a losing writer re-reads the head and tries again.
Bounded so a pathological contender fails loudly instead of spinning."""

MAX_DETAIL_CHARS: Final = 400
"""An event's detail says what happened in one breath — a refusal
reason, a halt reason, an error's message. The full account lives in the
stage's own records."""


class StageLogIntegrityError(RuntimeError):
    """The stored log contradicts itself: a gap, a broken chain, an event
    filed under an id it no longer derives, or unreadable JSON."""


class StageLogConflictError(RuntimeError):
    """An event file would be overwritten."""


class StageLogContentionError(RuntimeError):
    """An append lost its race for a sequence number too many times."""


@dataclass(frozen=True, slots=True)
class StageEvent:
    """One transition of one stage of one investigation.

    ``key`` is the idempotency key: the stage name and the content id of
    whatever determines its work — a directive for the analysis stages,
    the state a step begins from for experimentation. Because directives
    are content-addressed and derived deterministically from the config
    and the ids upstream, the key is the same in every process, which is
    what makes "has this already been done?" answerable after a crash.
    """

    investigation_id: str
    sequence: int
    stage: StageName
    status: StageStatus
    key: str
    subject_id: str
    """The directive (or state) the key derives from. Empty only when
    the stage never got as far as having one — a door that refused, or a
    stage skipped because the investigation ended upstream."""

    produced: tuple[tuple[str, str], ...] = ()
    """The ids this stage handed to the stages after it, as sorted
    name/id pairs. Replaying these in sequence order rebuilds the whole
    chain's state."""

    spend: StageSpend = ZERO_SPEND
    detail: str = ""
    previous_event_id: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.investigation_id.strip():
            raise ValueError("a stage event must name its investigation")
        if self.sequence < 0:
            raise ValueError(
                f"an event sequence starts at zero, got {self.sequence}"
            )
        if not self.key.strip():
            raise ValueError("a stage event must carry an idempotency key")
        if self.status is StageStatus.PENDING:
            raise ValueError(
                "pending is the absence of an event, never an event: "
                "writing one would make an empty log lie"
            )
        if self.status is StageStatus.RUNNING and self.produced:
            raise ValueError(
                "a running event predates the work; it cannot already "
                "name what the work produced"
            )
        if len(self.detail) > MAX_DETAIL_CHARS:
            raise ValueError(
                f"detail must be at most {MAX_DETAIL_CHARS} characters, "
                f"got {len(self.detail)}"
            )
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
            object.__setattr__(
                self,
                "id",
                content_id(
                    "sevt",
                    self.investigation_id,
                    self.sequence,
                    str(self.stage),
                    str(self.status),
                    self.key,
                    self.subject_id,
                    self.produced,
                    self.spend.model_calls,
                    self.spend.input_tokens,
                    self.spend.output_tokens,
                    self.detail,
                    self.previous_event_id,
                ),
            )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class StageLog:
    """The log of one investigation, addressed by a control root and an
    investigation id."""

    def __init__(self, root: Path | str, investigation_id: str) -> None:
        if not investigation_id.strip():
            raise ValueError("a stage log belongs to a named investigation")
        self._investigation_id = investigation_id
        self._directory = Path(root) / _LOGS / investigation_id
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def investigation_id(self) -> str:
        return self._investigation_id

    @property
    def directory(self) -> Path:
        return self._directory

    # -- reading ---------------------------------------------------------------

    def events(self) -> tuple[StageEvent, ...]:
        """Every event in sequence order, verified on the way out: ids
        re-derived, positions checked, the chain walked."""
        paths = sorted(self._directory.glob(f"*{_EVENT_SUFFIX}"))
        events: list[StageEvent] = []
        previous_id = ""
        for position, path in enumerate(paths):
            event = self._read(path)
            if event.sequence != position:
                raise StageLogIntegrityError(
                    f"log {self._investigation_id} jumps from sequence "
                    f"{position - 1} to {event.sequence}; an event is "
                    f"missing or misnamed"
                )
            if event.investigation_id != self._investigation_id:
                raise StageLogIntegrityError(
                    f"event {path.name} belongs to investigation "
                    f"{event.investigation_id}, not {self._investigation_id}"
                )
            if event.previous_event_id != previous_id:
                raise StageLogIntegrityError(
                    f"event {path.name} follows "
                    f"{event.previous_event_id or 'nothing'}, but the "
                    f"event before it is {previous_id or 'nothing'}"
                )
            events.append(event)
            previous_id = event.id
        return tuple(events)

    def terminal_for(self, key: str) -> StageEvent | None:
        """The last terminal event for one idempotency key, if any."""
        return next(
            (
                event
                for event in reversed(self.events())
                if event.key == key and event.is_terminal
            ),
            None,
        )

    def unfinished(self) -> StageEvent | None:
        """The ``RUNNING`` event no terminal event answered — the
        signature a crashed process leaves behind."""
        events = self.events()
        answered = {event.key for event in events if event.is_terminal}
        return next(
            (
                event
                for event in reversed(events)
                if event.status is StageStatus.RUNNING
                and event.key not in answered
            ),
            None,
        )

    def facts(self) -> ChainFacts:
        """Replay every succeeded stage's produced ids, in order."""
        facts = ChainFacts()
        for event in self.events():
            if event.status is StageStatus.SUCCEEDED:
                facts = facts.updated(event.produced)
        return facts

    def spend(self) -> StageSpend:
        """What the whole investigation has spent at the provider
        boundary, summed over the events that recorded any."""
        total = ZERO_SPEND
        for event in self.events():
            total = total.plus(event.spend)
        return total

    # -- writing ---------------------------------------------------------------

    def append(
        self,
        *,
        stage: StageName,
        status: StageStatus,
        key: str,
        subject_id: str = "",
        produced: Iterable[tuple[str, str]] = (),
        spend: StageSpend = ZERO_SPEND,
        detail: str = "",
    ) -> StageEvent:
        """Publish one event at the end of the log."""
        pairs = tuple(sorted(produced))
        for _ in range(_MAX_APPEND_ATTEMPTS):
            events = self.events()
            event = StageEvent(
                investigation_id=self._investigation_id,
                sequence=len(events),
                stage=stage,
                status=status,
                key=key,
                subject_id=subject_id,
                produced=pairs,
                spend=spend,
                detail=detail,
                previous_event_id=events[-1].id if events else "",
            )
            if self._publish(event):
                return event
            # Another writer took this sequence number; re-read the head
            # and rebuild against it.
        raise StageLogContentionError(
            f"log {self._investigation_id} lost {_MAX_APPEND_ATTEMPTS} races "
            f"for a sequence number; refusing to keep trying"
        )

    def _publish(self, event: StageEvent) -> bool:
        """Hard-link the event into place. ``False`` means the sequence
        was taken while this event was being built."""
        target = self._path(event.sequence)
        scratch = self._directory / f"{occurrence_id('escratch')}.tmp"
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
        return self._directory / f"{sequence:0{_SEQUENCE_DIGITS}d}{_EVENT_SUFFIX}"

    def _read(self, path: Path) -> StageEvent:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StageLogIntegrityError(
                f"stage event {path.name} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise StageLogIntegrityError(
                f"stage event {path.name} is not an object"
            )
        try:
            event = _event_from(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise StageLogIntegrityError(
                f"stage event {path.name} cannot be read: {exc}"
            ) from exc
        filed_as = payload.get("id")
        if filed_as != event.id:
            raise StageLogIntegrityError(
                f"stage event {path.name} claims id {filed_as!r} but "
                f"re-derives {event.id}; the file was edited"
            )
        return event


def _event_payload(event: StageEvent) -> dict[str, object]:
    return {
        "id": event.id,
        "investigation_id": event.investigation_id,
        "sequence": event.sequence,
        "stage": str(event.stage),
        "status": str(event.status),
        "key": event.key,
        "subject_id": event.subject_id,
        "produced": dict(event.produced),
        "spend": {
            "model_calls": event.spend.model_calls,
            "input_tokens": event.spend.input_tokens,
            "output_tokens": event.spend.output_tokens,
        },
        "detail": event.detail,
        "previous_event_id": event.previous_event_id,
    }


def _event_from(payload: dict[str, object]) -> StageEvent:
    spend = payload["spend"]
    if not isinstance(spend, dict):
        raise TypeError("spend must be an object")
    produced = payload["produced"]
    if not isinstance(produced, dict):
        raise TypeError("produced must be an object of name/id pairs")
    return StageEvent(
        investigation_id=_text(payload, "investigation_id"),
        sequence=_integer(payload, "sequence"),
        stage=StageName(_text(payload, "stage")),
        status=StageStatus(_text(payload, "status")),
        key=_text(payload, "key"),
        subject_id=_text(payload, "subject_id"),
        produced=tuple(sorted(_pairs(produced))),
        spend=StageSpend(
            model_calls=_integer(spend, "model_calls"),
            input_tokens=_integer(spend, "input_tokens"),
            output_tokens=_integer(spend, "output_tokens"),
        ),
        detail=_text(payload, "detail"),
        previous_event_id=_text(payload, "previous_event_id"),
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
