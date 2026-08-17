"""Typed scientific actions.

Research progress is modelled as a choice among typed actions rather than as a
fixed pipeline of stages.

A :class:`ResearchAction` is pure scientific *intent*: what to do, to what, and
why. It deliberately carries nothing else —

* cost and value estimates belong to :class:`~.decision.ActionUtility`,
  because an estimate is an opinion about the action, not part of it;
* execution status belongs to :class:`~.attempt.ActionAttempt`, because one
  intent may be attempted many times;
* the artefact the action produces is a typed domain object created by whoever
  performs it.

Actions are semantic objects with content identity: proposing the same action
twice is proposing the same action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ids import content_id


class ResearchActionType(StrEnum):
    SEARCH_LITERATURE = "search_literature"
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    REFINE_HYPOTHESIS = "refine_hypothesis"
    DERIVE_PREDICTION = "derive_prediction"
    DESIGN_EXPERIMENT = "design_experiment"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    RUN_EXPERIMENT = "run_experiment"
    ANALYZE = "analyze"
    REPLICATE = "replicate"
    TEST_BASELINE = "test_baseline"
    FALSIFY = "falsify"
    EXPLORE_ALTERNATIVE = "explore_alternative"
    SCALE_EXPERIMENT = "scale_experiment"
    SYNTHESIZE_FINDING = "synthesize_finding"
    ASSESS_CLAIM = "assess_claim"
    STOP_INVESTIGATION = "stop_investigation"


@dataclass(frozen=True, slots=True)
class ResearchAction:
    action_type: ResearchActionType
    rationale: str
    targets: tuple[str, ...] = ()
    """Ids of the domain objects this action operates on."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id("act", self.action_type, self.rationale, self.targets),
            )
