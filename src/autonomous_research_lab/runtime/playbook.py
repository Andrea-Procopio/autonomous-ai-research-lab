"""Research playbooks: a default prior over what to do next, never a pipeline.

A playbook encodes how research in a domain *usually* proceeds — for
empirical ML, roughly::

    establish viability -> establish baseline -> test central idea
    -> replicate promising result -> investigate mechanism
    -> validate against stronger baseline

Three deliberate limits keep this from becoming a stage machine:

* a playbook produces **advice** (a recommended emphasis plus a rationale),
  which the director is free to ignore;
* no code checks that a project "passed through" a stage — there is no stage
  state anywhere, only a reading of the current frontier;
* the whole mechanism is behind ``RuntimeConfig.playbook_enabled``, so its
  marginal value is measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .frontier import ResearchFrontier


@dataclass(frozen=True, slots=True)
class PlaybookStage:
    name: str
    goal: str


@dataclass(frozen=True, slots=True)
class PlaybookAdvice:
    """A recommendation, not an instruction."""

    stage: PlaybookStage
    rationale: str


class Playbook(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def stages(self) -> tuple[PlaybookStage, ...]: ...

    def advise(self, frontier: ResearchFrontier) -> PlaybookAdvice | None:
        """The stage this frontier most resembles, or ``None`` when the
        playbook has nothing useful to say."""
        ...


_VIABILITY = PlaybookStage(
    name="establish_viability",
    goal="get one end-to-end run producing real measurements",
)
_CENTRAL = PlaybookStage(
    name="test_central_idea",
    goal="run the experiment that bears on the central hypothesis",
)
_REPLICATE = PlaybookStage(
    name="replicate_promising_result",
    goal="repeat the protocol under its remaining declared seeds",
)
_MECHANISM = PlaybookStage(
    name="investigate_mechanism",
    goal="explain why results disagree before building on either",
)
_CONSOLIDATE = PlaybookStage(
    name="consolidate_findings",
    goal="turn measured evidence into assessed claims",
)


class EmpiricalMLPlaybook:
    """The one concrete playbook: a coarse reading of the frontier against
    the usual arc of empirical ML work. Rule-based and inspectable."""

    @property
    def name(self) -> str:
        return "empirical-ml:v1"

    @property
    def stages(self) -> tuple[PlaybookStage, ...]:
        return (_VIABILITY, _CENTRAL, _REPLICATE, _MECHANISM, _CONSOLIDATE)

    def advise(self, frontier: ResearchFrontier) -> PlaybookAdvice | None:
        if frontier.contradictions:
            return PlaybookAdvice(
                stage=_MECHANISM,
                rationale=frontier.contradictions[0].detail,
            )
        if not frontier.recent_results:
            return PlaybookAdvice(
                stage=_VIABILITY,
                rationale="nothing has been executed yet",
            )
        if frontier.replication_gaps:
            return PlaybookAdvice(
                stage=_REPLICATE,
                rationale=(
                    "a tested protocol still has declared seeds unused; "
                    "replication is available without new design work"
                ),
            )
        if frontier.pending_experiments or frontier.untested_predictions:
            return PlaybookAdvice(
                stage=_CENTRAL,
                rationale="designed or derivable tests have not been run",
            )
        if frontier.unsynthesized_evidence or frontier.unassessed_claims:
            return PlaybookAdvice(
                stage=_CONSOLIDATE,
                rationale="measured evidence has not been assessed",
            )
        return None
