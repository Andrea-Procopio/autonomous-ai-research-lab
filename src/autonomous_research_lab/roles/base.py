"""Specialized research roles.

A role is not a persona and not a prompt. It is the triple of

1. **objective** -- the utility it maximises;
2. **information access** -- what it is allowed to see;
3. **authority** -- which actions it may perform.

Two roles backed by the identical foundation model are still different agents
if those three differ. That is the separation worth having: a skeptic that is
rewarded for finding flaws behaves differently from a generator rewarded for
producing hypotheses, even with the same weights behind both.

Crucially, roles do not share a scalar reward. Each carries its own
:class:`UtilityFunction`. The interface is intentionally weak about how utility
is computed -- a hand-written heuristic, a model evaluation, and a learned
scorer all satisfy it -- because which of those is right is an open research
question, not a settled design.

Execution (``perform``) is not part of this interface yet: performing most
actions requires a model provider, which is out of scope until the contracts
below have been exercised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ..core.actions import ResearchAction, ResearchActionType
from ..core.state import ResearchState
from ..core.types import freeze_mapping


class RoleName(StrEnum):
    RESEARCH_DIRECTOR = "research_director"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    LITERATURE_RESEARCHER = "literature_researcher"
    SKEPTIC = "skeptic"
    METHODOLOGIST = "methodologist"
    RESEARCH_ENGINEER = "research_engineer"
    EXPERIMENT_DESIGNER = "experiment_designer"
    STATISTICIAN = "statistician"
    RESULT_ANALYST = "result_analyst"
    EVIDENCE_VERIFIER = "evidence_verifier"
    SCIENTIFIC_REVIEWER = "scientific_reviewer"
    PAPER_WRITER = "paper_writer"


@dataclass(frozen=True, slots=True)
class UtilityScore:
    value: float
    components: Mapping[str, float] = field(default_factory=dict)
    """Named contributions to ``value`` -- novelty, falsifiability, expected
    information gain, and so on. Kept separate from the scalar so that a
    decision can be audited rather than merely ranked."""

    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", freeze_mapping(self.components))


class UtilityFunction(Protocol):
    def __call__(
        self, state: ResearchState, action: ResearchAction
    ) -> UtilityScore: ...


@dataclass(frozen=True, slots=True)
class WeightedUtility:
    """A utility built from named component scorers and weights.

    Included because it makes the "each role has its own objective" claim
    concrete without pretending to know the right components. The weights are
    an explicit, editable statement of what a role is being asked to maximise.
    """

    weights: Mapping[str, float]
    scorers: Mapping[str, UtilityFunction]

    def __call__(self, state: ResearchState, action: ResearchAction) -> UtilityScore:
        components: dict[str, float] = {}
        total = 0.0
        for name, weight in self.weights.items():
            scorer = self.scorers.get(name)
            if scorer is None:
                continue
            component = scorer(state, action).value
            components[name] = component
            total += weight * component
        return UtilityScore(value=total, components=components)


class ResearchRole(ABC):
    @property
    @abstractmethod
    def name(self) -> RoleName: ...

    @property
    @abstractmethod
    def supported_actions(self) -> frozenset[ResearchActionType]:
        """The actions this role is authorised to perform."""

    @abstractmethod
    def utility(self, state: ResearchState, action: ResearchAction) -> UtilityScore:
        """This role's own valuation of taking ``action`` in ``state``."""

    def can_perform(self, action_type: ResearchActionType) -> bool:
        return action_type in self.supported_actions
