"""Specialized research roles.

A role is not a persona and not a prompt. It is the quadruple of

1. **objective** — what it is asked to accomplish;
2. **information set** — the explicit view of state it receives;
3. **allowed actions/tools** — which action types it may perform;
4. **output contract** — which proposal kinds it may return.

Two roles backed by the identical foundation model are still different agents
if those differ: a skeptic rewarded for finding flaws behaves differently
from a generator rewarded for producing hypotheses, even with the same
weights behind both.

**Architectural invariant: roles never mutate ResearchState.** A role reads a
:class:`RoleContext` and produces :mod:`~autonomous_research_lab.core.proposals`
— typed, attributable requests — which only the transition layer validates and
commits. This is enforced structurally by ``tests/test_layering.py``: no module
in this package may call a state mutator or import the transition layer. The
payoff is provenance (every change names its proposer), safe search branching,
and one place for conflict resolution when multiple agents propose
concurrently.

Two value concepts, deliberately not one
----------------------------------------

:class:`~autonomous_research_lab.core.decision.ActionUtility` answers
*"should the lab perform this action?"* — ``U(a | state)``, an estimate of
scientific value, owned by the utility evaluator and consumed by the search
policy.

:class:`RoleSuitability` answers *"who should perform this selected
action?"* — approximately ``P(role succeeds | action, state)``, an estimate of
fit between a role and work already chosen. It deliberately does not use the
word *utility*: it expresses no opinion about whether the action is worth
performing, and letting it leak into action selection would collapse "what is
valuable" into "what our current roles happen to be good at".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ..core.actions import ResearchAction, ResearchActionType
from ..core.assessment import EpistemicAssessment
from ..core.budget import NO_COST, ResourceCost
from ..core.claim import Claim, EvidenceLink
from ..core.evidence import Evidence
from ..core.experiment import ExperimentResult, ExperimentSpec
from ..core.hypothesis import Hypothesis
from ..core.ids import occurrence_id
from ..core.prediction import Prediction, PredictionTest
from ..core.proposals import Proposal, ProposalKind, kind_of
from ..core.question import ResearchQuestion
from ..core.replication import ReplicationGroup
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
class RoleSuitability:
    """How well-suited a role is to perform a particular action — the routing
    quantity, never the action-selection quantity (see module docstring)."""

    value: float
    components: Mapping[str, float] = field(default_factory=dict)
    """Named contributions to ``value``, kept separate from the scalar so that
    an assignment can be audited rather than merely ranked."""

    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", freeze_mapping(self.components))


class SuitabilityFunction(Protocol):
    def __call__(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability: ...


@dataclass(frozen=True, slots=True)
class WeightedSuitability:
    """A suitability estimate built from named component scorers and weights —
    an explicit, editable statement of what makes a role fit for work."""

    weights: Mapping[str, float]
    scorers: Mapping[str, SuitabilityFunction]

    def __call__(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        components: dict[str, float] = {}
        total = 0.0
        for name, weight in self.weights.items():
            scorer = self.scorers.get(name)
            if scorer is None:
                continue
            component = scorer(state, action).value
            components[name] = component
            total += weight * component
        return RoleSuitability(value=total, components=components)


@dataclass(frozen=True, slots=True)
class RoleContext:
    """The explicit information set a role receives — a projection built by
    the orchestrator, never the raw ``ResearchState``.

    Every field defaults to empty: the orchestrator includes exactly what the
    invocation needs, and what a role was shown is thereby recorded rather
    than implied. Illustrative projections (built per-role, later):

    * a hypothesis researcher sees the question, current hypotheses, and the
      important negative findings;
    * a skeptic sees one hypothesis, its predictions, the supporting evidence,
      and the alternative explanations on record;
    * a research engineer sees an experiment spec and execution constraints;
    * a statistician sees raw results, the experiment design, the replication
      group, and the claims at stake.

    ``notes`` carries orchestrator guidance that is contextual rather than
    structural ("prior attempt failed with X"). It is not a prompt: rendering
    a context for a model is a provider-boundary concern, out of scope here.
    """

    objective: str = ""
    questions: tuple[ResearchQuestion, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    prediction_tests: tuple[PredictionTest, ...] = ()
    experiments: tuple[ExperimentSpec, ...] = ()
    results: tuple[ExperimentResult, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[Claim, ...] = ()
    evidence_links: tuple[EvidenceLink, ...] = ()
    assessments: tuple[EpistemicAssessment, ...] = ()
    replication_groups: tuple[ReplicationGroup, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoleInvocation:
    """The contract under which a role receives one piece of work.

    Occurrence identity: invoking the same role on the same assignment twice
    is two invocations. The invocation pins, as data, everything that
    ``role = objective + information set + allowed actions + output
    contract`` promises — so that what a role could see and do is auditable
    per invocation, not a property of prose."""

    role: RoleName
    assignment: ResearchAction
    context: RoleContext
    allowed_actions: frozenset[ResearchActionType]
    expected_output: frozenset[ProposalKind]
    budget: ResourceCost = NO_COST
    """The most this invocation may spend, not a program-level budget."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.assignment.action_type not in self.allowed_actions:
            raise ValueError(
                f"invocation assigns {self.assignment.action_type}, which is "
                f"not among its allowed actions"
            )
        if not self.id:
            object.__setattr__(self, "id", occurrence_id("inv"))

    def permits(self, proposal: Proposal) -> bool:
        """Whether ``proposal`` is within this invocation's output contract."""
        return kind_of(proposal) in self.expected_output


class ResearchRole(ABC):
    @property
    @abstractmethod
    def name(self) -> RoleName: ...

    @property
    @abstractmethod
    def supported_actions(self) -> frozenset[ResearchActionType]:
        """The actions this role is authorised to perform."""

    @abstractmethod
    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        """This role's estimated fit for performing ``action`` in ``state``."""

    @abstractmethod
    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        """Do the assigned work and return proposals — never a state.

        Every proposal returned must satisfy ``invocation.permits``. Concrete
        implementations need a model provider and are deliberately absent
        until one exists; implementations without one raise
        ``NotImplementedError``."""

    def can_perform(self, action_type: ResearchActionType) -> bool:
        return action_type in self.supported_actions
