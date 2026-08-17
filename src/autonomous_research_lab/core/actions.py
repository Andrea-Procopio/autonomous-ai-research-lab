"""Typed scientific actions.

Research progress is modelled as a choice among typed actions rather than as a
fixed pipeline of stages. An orchestrator enumerates candidate actions; a search
policy chooses among them; a role executes the chosen one.

A :class:`ResearchAction` is a *decision record* -- what to do, to what, why,
and at what expected cost. It deliberately does not carry the artefact the
action produces; that artefact is a typed domain object created by whoever
performs the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .budget import NO_COST, ResourceCost
from .ids import content_id


class ResearchActionType(StrEnum):
    SEARCH_LITERATURE = "search_literature"
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    REFINE_HYPOTHESIS = "refine_hypothesis"
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
    STOP_INVESTIGATION = "stop_investigation"


@dataclass(frozen=True, slots=True)
class ResearchAction:
    action_type: ResearchActionType
    rationale: str
    targets: tuple[str, ...] = ()
    """Ids of the domain objects this action operates on."""

    estimated_cost: ResourceCost = NO_COST
    expected_information_gain: float | None = None
    """Expected reduction in scientific uncertainty, in whatever units the
    scoring policy defines. ``None`` means "not estimated"; policies must not
    silently read that as zero value."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "act", self.action_type, self.rationale, self.targets
                ),
            )
