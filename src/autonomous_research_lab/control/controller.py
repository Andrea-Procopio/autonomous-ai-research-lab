"""Walking the chain: one stage at a time, once each, resumably.

The loop is deliberately dull. For every stage, in order:

1. **Plan.** Derive the stage's directive from the config and the ids
   already produced. The directive's content id is the idempotency key.
2. **Remember.** If the log already holds a succeeded event for that
   key, adopt what it produced and move on. Nothing is re-run and
   nothing is re-paid.
3. **Reconcile.** Otherwise ask the stage's own store whether the work
   is there anyway. It will be exactly when a process died between the
   side effect and the record of it, and the honest response is to write
   the missing event rather than to buy the work twice.
4. **Announce, act, record.** Write ``RUNNING``, call the stage, write
   the terminal event. In that order, always, so a crash is legible.

Nothing retries. A refusal or a failure is written down and the walk
stops; ``resume`` re-attempts that stage and only that stage.

Three ways for a walk to end well, and the difference matters. It can
run out of stages (*completed*), reach the stage the operator asked it
to stop after (*stopped*), or hear an honest scientific no — no adequate
map, no candidate worth proposing, no eligible candidate, a runtime that
halted (*ended*). The third is not a malfunction. It is the outcome the
whole architecture exists to be able to reach, and the controller
records it by marking the stages that will now never run as ``SKIPPED``
— "did not happen, and never will", which is a different fact from "not
yet".

The config a resumed walk uses is read from the investigation's own
record, never from the file it came from. An operator who edits the
file and resumes gets the run they started, not a hybrid of two.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..core.ids import occurrence_id
from .chain import Stage, StageContext, Stores, default_chain
from .config import RunConfig, parse_config
from .events import StageEvent, StageLog
from .investigation import Investigation, InvestigationStore
from .lab import DefaultLab, Lab
from .stage import (
    CHAIN_ORDER,
    ChainFacts,
    Fact,
    StageName,
    StageSpend,
    StageStatus,
)

_CONTROL = "control"

_STATE_FACTS = frozenset(
    {str(Fact.STATE_ID), str(Fact.FUNDED_STATE_ID), str(Fact.ADMITTED_STATE_ID)}
)


class ControllerError(RuntimeError):
    """The controller cannot proceed: an unknown investigation, a config
    that vanished, or a stage that claims progress it did not make."""


class Outcome(StrEnum):
    """How one walk ended."""

    COMPLETED = "completed"
    """Every stage ran. For a chain ending in experimentation this means
    the runtime halted on its own terms."""

    STOPPED = "stopped"
    """The stage the operator asked to stop after has finished. Resuming
    continues past it."""

    ENDED = "ended"
    """An honest scientific no. Nothing further to try, nothing broken."""

    REFUSED = "refused"
    """A door refused a stage's preconditions."""

    FAILED = "failed"
    """A stage raised. The cause is on the event; the fix is the
    operator's."""


@dataclass(frozen=True, slots=True)
class WalkResult:
    """What one walk did, and where the investigation now stands."""

    investigation_id: str
    outcome: Outcome
    detail: str
    facts: ChainFacts
    events: tuple[StageEvent, ...]

    @property
    def ok(self) -> bool:
        """True when the walk ended on its own terms rather than on a
        refusal or a fault. An honest no counts as ok."""
        return self.outcome in (
            Outcome.COMPLETED,
            Outcome.STOPPED,
            Outcome.ENDED,
        )


@dataclass(frozen=True, slots=True)
class StageLine:
    """One row of ``arl status``."""

    stage: StageName
    status: StageStatus
    detail: str
    spend: StageSpend


@dataclass(frozen=True, slots=True)
class StatusReport:
    investigation: Investigation
    lines: tuple[StageLine, ...]
    facts: ChainFacts
    spend: StageSpend


@dataclass
class Controller:
    """One run root, one chain, and the log that remembers both."""

    root: Path
    chain: tuple[Stage, ...] = field(default_factory=default_chain)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._stores = Stores.under(self.root)
        self._investigations = InvestigationStore(self.root / _CONTROL)

    @property
    def stores(self) -> Stores:
        return self._stores

    @property
    def investigations(self) -> InvestigationStore:
        return self._investigations

    # -- starting and resuming -------------------------------------------------

    def begin(self, payload: Mapping[str, object]) -> Investigation:
        """Record one config and the investigation that will use it.

        Parsing happens first, so a config that cannot produce a legal
        run leaves nothing behind at all.
        """
        config = parse_config(payload)
        config_id = self._investigations.record_config(payload)
        return self._investigations.record(
            Investigation(
                investigation_id=occurrence_id("inv"),
                config_id=config_id,
                label=config.label,
                stop_after=(
                    str(config.stop_after)
                    if config.stop_after is not None
                    else ""
                ),
            )
        )

    def run(
        self,
        payload: Mapping[str, object],
        *,
        lab: Lab | None = None,
        stop_after: StageName | None = None,
    ) -> WalkResult:
        return self.walk(
            self.begin(payload), lab=lab, stop_after=stop_after
        )

    def resume(
        self,
        investigation_id: str,
        *,
        lab: Lab | None = None,
        stop_after: StageName | None = None,
    ) -> WalkResult:
        return self.walk(
            self._require(investigation_id), lab=lab, stop_after=stop_after
        )

    # -- the walk --------------------------------------------------------------

    def walk(
        self,
        investigation: Investigation,
        *,
        lab: Lab | None = None,
        stop_after: StageName | None = None,
    ) -> WalkResult:
        """Walk as far as this investigation can go.

        ``stop_after`` is this walk's brake and is not recorded: the
        operator who asked for it can resume past it. The scope the
        config declared is recorded, and resuming does not pass that —
        an investigation meant to reach a funded run and stop is not
        talked into experimenting by being resumed.
        """
        config = self._config_of(investigation)
        log = self._investigations.log_for(investigation.investigation_id)
        context = StageContext(
            stores=self._stores,
            config=config,
            lab=lab if lab is not None else DefaultLab(),
            facts=log.facts(),
            known_states=_states_mentioned(log),
        )
        ended = _already_ended(log)
        if ended is not None:
            return self._result(investigation, log, Outcome.ENDED, ended.detail)

        declared = (
            StageName(investigation.stop_after)
            if investigation.stop_after
            else None
        )
        halt_after = stop_after if stop_after is not None else declared
        for stage in self.chain:
            context, outcome, detail = self._walk_stage(stage, context, log)
            if outcome is not None:
                if outcome is Outcome.ENDED:
                    _skip_the_rest(log, stage.name, detail)
                return self._result(investigation, log, outcome, detail)
            if halt_after is not None and stage.name is halt_after:
                return self._result(
                    investigation,
                    log,
                    Outcome.STOPPED,
                    f"stopped after {stage.name} as asked",
                )
        return self._result(
            investigation, log, Outcome.COMPLETED, "every stage is done"
        )

    def _walk_stage(
        self, stage: Stage, context: StageContext, log: StageLog
    ) -> tuple[StageContext, Outcome | None, str]:
        """Run one stage to exhaustion. ``None`` means carry on."""
        seen: set[str] = set()
        expecting_more = False
        executions = 0
        while True:
            plan = stage.plan(context)
            if plan.key in seen:
                if expecting_more:
                    raise ControllerError(
                        f"stage {stage.name} asked to continue but planned "
                        f"the same work again ({plan.key}); a repeat with no "
                        f"progress would loop forever"
                    )
                return context, None, ""
            seen.add(plan.key)

            recorded = log.terminal_for(plan.key)
            if recorded is not None and recorded.status is StageStatus.SUCCEEDED:
                context = context.with_produced(recorded.produced)
                expecting_more = False
                continue

            reconciled = stage.completed(context, plan)
            if reconciled is not None:
                log.append(
                    stage=stage.name,
                    status=StageStatus.SUCCEEDED,
                    key=plan.key,
                    subject_id=plan.subject_id,
                    produced=reconciled.produced,
                    spend=reconciled.spend,
                    detail=f"reconciled: {reconciled.detail}"[:400],
                )
                context = context.with_produced(reconciled.produced)
                if reconciled.ends_investigation:
                    return context, Outcome.ENDED, reconciled.ends_investigation
                if not reconciled.repeat:
                    return context, None, ""
                expecting_more = True
                continue

            if executions >= stage.limit(context.config):
                return (
                    context,
                    Outcome.STOPPED,
                    f"{stage.name} reached its limit of "
                    f"{stage.limit(context.config)} attempt(s)",
                )

            log.append(
                stage=stage.name,
                status=StageStatus.RUNNING,
                key=plan.key,
                subject_id=plan.subject_id,
            )
            try:
                outcome = stage.execute(context)
            except stage.refusals() as refusal:
                log.append(
                    stage=stage.name,
                    status=StageStatus.REFUSED,
                    key=plan.key,
                    subject_id=plan.subject_id,
                    detail=str(refusal)[:400],
                )
                return context, Outcome.REFUSED, str(refusal)
            except Exception as error:  # recorded, never re-raised
                log.append(
                    stage=stage.name,
                    status=StageStatus.FAILED,
                    key=plan.key,
                    subject_id=plan.subject_id,
                    detail=f"{type(error).__name__}: {error}"[:400],
                )
                return context, Outcome.FAILED, f"{type(error).__name__}: {error}"

            executions += 1
            log.append(
                stage=stage.name,
                status=StageStatus.SUCCEEDED,
                key=plan.key,
                subject_id=plan.subject_id,
                produced=outcome.produced,
                spend=outcome.spend,
                detail=outcome.detail[:400],
            )
            context = context.with_produced(outcome.produced)
            if outcome.ends_investigation:
                return context, Outcome.ENDED, outcome.ends_investigation
            if not outcome.repeat:
                return context, None, ""
            expecting_more = True

    # -- reporting -------------------------------------------------------------

    def status(self, investigation_id: str) -> StatusReport:
        investigation = self._require(investigation_id)
        log = self._investigations.log_for(investigation_id)
        events = log.events()
        lines: list[StageLine] = []
        for stage in CHAIN_ORDER:
            for_stage = [event for event in events if event.stage is stage]
            if not for_stage:
                lines.append(
                    StageLine(stage, StageStatus.PENDING, "", StageSpend())
                )
                continue
            last = for_stage[-1]
            spend = StageSpend()
            for event in for_stage:
                spend = spend.plus(event.spend)
            lines.append(StageLine(stage, last.status, last.detail, spend))
        return StatusReport(
            investigation=investigation,
            lines=tuple(lines),
            facts=log.facts(),
            spend=log.spend(),
        )

    # -- internals -------------------------------------------------------------

    def _require(self, investigation_id: str) -> Investigation:
        investigation = self._investigations.get(investigation_id)
        if investigation is None:
            raise ControllerError(
                f"no investigation {investigation_id} under {self.root}"
            )
        return investigation

    def _config_of(self, investigation: Investigation) -> RunConfig:
        payload = self._investigations.get_config(investigation.config_id)
        if payload is None:
            raise ControllerError(
                f"investigation {investigation.investigation_id} names config "
                f"{investigation.config_id}, which is not recorded; the run "
                f"cannot be resumed without what it was asked to do"
            )
        return parse_config(payload)

    def _result(
        self,
        investigation: Investigation,
        log: StageLog,
        outcome: Outcome,
        detail: str,
    ) -> WalkResult:
        return WalkResult(
            investigation_id=investigation.investigation_id,
            outcome=outcome,
            detail=detail,
            facts=log.facts(),
            events=log.events(),
        )


def _states_mentioned(log: StageLog) -> frozenset[str]:
    """Every state id the log has ever produced, succeeded or not.

    "Ever", not "currently": a state a failed step produced is still a
    state this investigation knows about, and treating it as an orphan
    afterwards would adopt the same step twice.
    """
    return frozenset(
        value
        for event in log.events()
        for name, value in event.produced
        if name in _STATE_FACTS
    )


def _already_ended(log: StageLog) -> StageEvent | None:
    """An investigation that reached an honest no left skipped stages
    behind it; there is nothing to walk."""
    return next(
        (
            event
            for event in log.events()
            if event.status is StageStatus.SKIPPED
        ),
        None,
    )


def _skip_the_rest(log: StageLog, after: StageName, reason: str) -> None:
    """Record that the stages downstream of an honest no will never
    run — a different fact from not having run yet."""
    # A no at the last stage marks that stage itself: there is nothing
    # after it to mark, and "no further steps will be taken" still needs
    # recording or a resumed walk would start another one.
    remaining = CHAIN_ORDER[CHAIN_ORDER.index(after) + 1 :] or (after,)
    for stage in remaining:
        log.append(
            stage=stage,
            status=StageStatus.SKIPPED,
            key=f"{stage}:skipped",
            detail=reason[:400],
        )


__all__ = [
    "Controller",
    "ControllerError",
    "Outcome",
    "StageLine",
    "StatusReport",
    "WalkResult",
]
