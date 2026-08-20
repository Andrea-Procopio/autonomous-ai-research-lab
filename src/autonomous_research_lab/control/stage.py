"""The vocabulary of a walked chain: stages, statuses, spend, and facts.

Three small types, each answering one question about a stage.

**Which stage** — :class:`StageName`. The chain is fixed and ordered, and
:data:`CHAIN_ORDER` is that order. A DAG with one path is a list, and
calling it a list keeps the controller readable; the day a stage forks,
the order becomes a graph and this constant becomes its topological
sort.

**What happened to it** — :class:`StageStatus`. The six the audit asked
for, with two distinctions worth stating outright. ``PENDING`` is never
written: a stage nobody has attempted has no event, and inventing a
record for the absence of one would make an empty log lie about what it
knows. ``REFUSED`` means a *door* said no — an inadequate map, a lineage
that will not verify — before any model call and any spend; an honest
scientific no, such as a selection with no eligible candidate, is a
``SUCCEEDED`` stage that happens to end the investigation. Confusing the
two would file the system's most valuable outcome as a malfunction.

**What it produced** — :class:`ChainFacts`. Every id a later stage needs
from an earlier one, in one flat, string-keyed mapping. Flat on purpose:
the controller rebuilds it after a crash by replaying the event log, and
a structure it can rebuild generically is a structure that cannot drift
from what the log actually recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class StageName(StrEnum):
    """The stages of the chain, in the order they must happen."""

    MAPPING = "mapping"
    IDEATION = "ideation"
    PRIOR_ART = "prior_art"
    SELECTION = "selection"
    ADMISSION = "admission"
    FUNDING = "funding"
    EXPERIMENTATION = "experimentation"


CHAIN_ORDER: Final[tuple[StageName, ...]] = (
    StageName.MAPPING,
    StageName.IDEATION,
    StageName.PRIOR_ART,
    StageName.SELECTION,
    StageName.ADMISSION,
    StageName.FUNDING,
    StageName.EXPERIMENTATION,
)


class StageStatus(StrEnum):
    """What the log says about one attempt at one stage."""

    PENDING = "pending"
    """Never written. The absence of an event *is* pending, and the
    status exists so a report can name that absence."""

    RUNNING = "running"
    """Written before the side effect, so a process that dies mid-stage
    leaves a visible claim rather than a silent gap."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"

    REFUSED = "refused"
    """A door refused the stage's preconditions. No call, no spend, and
    no reason to retry without changing something."""

    SKIPPED = "skipped"
    """The investigation ended upstream, so this stage will never be
    attempted for it. Recorded rather than left pending: "did not
    happen, and never will" is a different fact from "not yet"."""


TERMINAL_STATUSES: Final = frozenset(
    {
        StageStatus.SUCCEEDED,
        StageStatus.FAILED,
        StageStatus.REFUSED,
        StageStatus.SKIPPED,
    }
)
"""The statuses that end an attempt. Anything else is in flight."""


@dataclass(frozen=True, slots=True)
class StageSpend:
    """What one stage cost at the provider boundary.

    Model calls and tokens only. Wall clock and dollars belong to the
    funded run's ledger, which starts existing at the funding stage; the
    analysis stages before it spend tokens against their own directive
    caps and nothing else.
    """

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("model_calls", self.model_calls),
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative, got {value}")

    @property
    def is_zero(self) -> bool:
        return not (self.model_calls or self.input_tokens or self.output_tokens)

    def plus(self, other: StageSpend) -> StageSpend:
        return StageSpend(
            model_calls=self.model_calls + other.model_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


ZERO_SPEND: Final = StageSpend()


class Fact(StrEnum):
    """The ids one stage hands to the stages after it.

    Only identifiers live here. A fact is a pointer into a durable store,
    never a summary of what the record says: the controller must have no
    opinion it could hold instead of reading.
    """

    MAP_RUN_RECORD_ID = "map_run_record_id"
    ASSESSMENT_ID = "assessment_id"
    SNAPSHOT_ID = "snapshot_id"
    IDEATION_RUN_RECORD_ID = "ideation_run_record_id"
    PRIOR_ART_RUN_RECORD_ID = "prior_art_run_record_id"
    SELECTION_RUN_RECORD_ID = "selection_run_record_id"
    ADMISSION_RECORD_ID = "admission_record_id"
    ADMITTED_STATE_ID = "admitted_state_id"
    RUN_ID = "run_id"
    FUNDED_STATE_ID = "funded_state_id"
    STATE_ID = "state_id"
    """The head of the funded run's snapshot chain: the state the next
    experimentation step starts from."""


class MissingFactError(LookupError):
    """A stage asked for an id no earlier stage recorded."""


@dataclass(frozen=True, slots=True)
class ChainFacts:
    """The ids produced so far, replayable from the event log alone."""

    entries: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        keys = [key for key, _ in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("a fact is recorded once per investigation")
        for key, value in self.entries:
            if not key.strip() or not value.strip():
                raise ValueError("a fact needs both a name and an id")

    @classmethod
    def of(cls, mapping: Mapping[str, str]) -> ChainFacts:
        return cls(tuple(sorted(mapping.items())))

    def as_mapping(self) -> Mapping[str, str]:
        return dict(self.entries)

    def get(self, fact: Fact) -> str | None:
        return next(
            (value for key, value in self.entries if key == str(fact)), None
        )

    def require(self, fact: Fact) -> str:
        value = self.get(fact)
        if value is None:
            raise MissingFactError(
                f"no stage has recorded {fact}; the chain cannot continue "
                f"without it"
            )
        return value

    def updated(self, produced: Iterable[tuple[str, str]]) -> ChainFacts:
        """Return the facts with ``produced`` layered on top.

        Later wins, because replaying the log means replaying a stage's
        re-run over its first attempt.
        """
        merged = dict(self.entries)
        merged.update(produced)
        return ChainFacts.of(merged)


NO_FACTS: Final = ChainFacts()
"""Nothing produced yet — where every investigation starts."""
