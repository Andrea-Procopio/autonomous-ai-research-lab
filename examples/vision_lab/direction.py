"""The composite seat: deterministic follow-through first, then the planner.

Task 7A bootstrapped the funded state deterministically because the
planner's gate demands cited admissible evidence and a fresh funded
state has none. This module is the hand-off 7A promised: one director
and one seat role that put the deterministic machinery first and give
the model the floor exactly when it has something legitimate to say —
verified evidence on the record, and nothing mechanical left to do.

The routing rules, stated once and pinned by tests:

* Structural and analytical work the rule-based director offers —
  design, synthesis, assessment, hypothesis or prediction derivation —
  is returned as-is. The deterministic follow-through always runs
  first.
* Execution work (run, replicate) is delegated to the
  :class:`PlanningDirector` exactly when a planning record owns it, so
  the store's dispatch bookkeeping happens for planner-created work
  and stays out of bootstrap work. A replicate is delegated only when
  a REPLICATE decision exists for the target: the planning director
  has no branch for a bare replication gap and would fall through to
  an unintended, billed consultation.
* A rule-based STOP becomes a planner consultation — or the dispatch
  of a planner decision already on record — when verified findings
  exist and no earlier consultation ended in terminal rejection.
  The rejection guard reads the planning store, not the frontier: a
  frontier's failed-attempt view goes blind to a failed consultation
  the moment any other consultation has succeeded.

Two invariants the rules exist to keep. The rule-based director is
consulted first because it is pure; the planning director is called at
most once per step and never speculatively, because its deliberation
writes dispatch bookkeeping into the planning store — a consultation
that was then discarded would bill work nobody dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autonomous_research_lab.core.actions import ResearchAction, ResearchActionType
from autonomous_research_lab.core.proposals import Proposal
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.director import (
    Deliberation,
    RuleBasedFrontierDirector,
)
from autonomous_research_lab.orchestration.planning import PlanningDirector
from autonomous_research_lab.orchestration.synthesis import SynthesisReview
from autonomous_research_lab.roles.base import (
    ResearchRole,
    RoleInvocation,
    RoleName,
    RoleSuitability,
)
from autonomous_research_lab.roles.planner import ModelBackedPlanner
from autonomous_research_lab.runtime.escalation import ReasoningTier
from autonomous_research_lab.runtime.frontier import ResearchFrontier
from autonomous_research_lab.runtime.planning_store import (
    PlanningAction,
    PlanningStore,
)
from autonomous_research_lab.runtime.playbook import PlaybookAdvice

from .science import VisionScientist

_FOLLOW_THROUGH = frozenset(
    {
        ResearchActionType.GENERATE_HYPOTHESIS,
        ResearchActionType.REFINE_HYPOTHESIS,
        ResearchActionType.DERIVE_PREDICTION,
        ResearchActionType.DESIGN_EXPERIMENT,
        ResearchActionType.SYNTHESIZE_FINDING,
        ResearchActionType.ASSESS_CLAIM,
        ResearchActionType.ANALYZE,
    }
)

_EXECUTION_DECISIONS = frozenset(
    {PlanningAction.NEW_EXPERIMENT, PlanningAction.ABLATION}
)


@dataclass
class VisionDirector:
    """Deterministic frontier work first; the planner when it is earned."""

    plans: PlanningStore
    rule_based: RuleBasedFrontierDirector = field(
        default_factory=RuleBasedFrontierDirector
    )
    planning: PlanningDirector | None = None

    def __post_init__(self) -> None:
        if self.planning is None:
            self.planning = PlanningDirector(plans=self.plans)

    @property
    def name(self) -> str:
        return "frontier-director:vision-composite:v1"

    def deliberate(
        self,
        frontier: ResearchFrontier,
        *,
        advice: PlaybookAdvice | None = None,
        tier: ReasoningTier = ReasoningTier.ROUTINE,
        max_candidates: int = 3,
    ) -> Deliberation:
        assert self.planning is not None
        ruled = self.rule_based.deliberate(
            frontier, advice=advice, tier=tier, max_candidates=max_candidates
        )
        selected = ruled.selected
        if selected is None:
            return ruled
        action = selected.action.action_type
        if action in _FOLLOW_THROUGH:
            return ruled
        if action is ResearchActionType.RUN_EXPERIMENT:
            if self._owned_by_a_decision(selected.action.targets):
                return self._delegate(frontier, advice, tier, max_candidates)
            return ruled
        if action is ResearchActionType.REPLICATE:
            if self._has_replicate_decision(selected.action.targets):
                return self._delegate(frontier, advice, tier, max_candidates)
            return ruled
        if action is ResearchActionType.STOP_INVESTIGATION:
            if self._planner_may_speak(frontier):
                return self._delegate(frontier, advice, tier, max_candidates)
            return ruled
        return ruled

    def synthesize(
        self,
        frontier: ResearchFrontier,
        *,
        tier: ReasoningTier = ReasoningTier.ROUTINE,
    ) -> SynthesisReview:
        return self.rule_based.synthesize(frontier, tier=tier)

    def _delegate(
        self,
        frontier: ResearchFrontier,
        advice: PlaybookAdvice | None,
        tier: ReasoningTier,
        max_candidates: int,
    ) -> Deliberation:
        """The one planning-director call a step may make, returned
        verbatim — its deliberation writes dispatch bookkeeping, so a
        discarded or repeated call would bill work nobody dispatched."""
        assert self.planning is not None
        return self.planning.deliberate(
            frontier, advice=advice, tier=tier, max_candidates=max_candidates
        )

    def _owned_by_a_decision(self, targets: tuple[str, ...]) -> bool:
        if not targets:
            return False
        return any(
            record.spec_id == targets[0]
            and record.action in _EXECUTION_DECISIONS
            for record in self.plans.records()
        )

    def _has_replicate_decision(self, targets: tuple[str, ...]) -> bool:
        if not targets:
            return False
        return any(
            record.spec_id == targets[0]
            and record.action is PlanningAction.REPLICATE
            for record in self.plans.records()
        )

    def _planner_may_speak(self, frontier: ResearchFrontier) -> bool:
        """Verified findings exist, and no consultation ended in terminal
        rejection.

        The rejection guard is the store's, not the frontier's: a
        rejected invocation that never became a record is a consultation
        the gate refused twice, and repeating that billed work cannot
        improve it. The frontier's failed-attempt view stops seeing such
        a failure once any other consultation succeeds, which is exactly
        the moment the guard must not go blind.
        """
        if not frontier.best_findings:
            return False
        if any(
            attempt.action.action_type
            is ResearchActionType.PLAN_NEXT_ACTION
            for attempt in frontier.failed_attempts
        ):
            return False
        recorded = {record.invocation_id for record in self.plans.records()}
        rejected = {
            str(entry["invocation_id"]) for entry in self.plans.rejected()
        }
        return not (rejected - recorded)


class VisionDirectorRole(ResearchRole):
    """The one RESEARCH_DIRECTOR seat, two trusted halves.

    The deterministic scientist designs and synthesizes; the model-backed
    planner answers exactly one action, the consultation. One seat
    because routing is the director's job and the loop maps one role per
    seat; two halves because a planner that also authored designs would
    be a model deciding what the bootstrap deliberately keeps
    deterministic.
    """

    def __init__(
        self, scientist: VisionScientist, planner: ModelBackedPlanner
    ) -> None:
        self._scientist = scientist
        self._planner = planner

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_DIRECTOR

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return (
            self._scientist.supported_actions
            | self._planner.supported_actions
        )

    def suitability(
        self, state: ResearchState, action: ResearchAction
    ) -> RoleSuitability:
        del state, action
        return RoleSuitability(value=1.0)

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        if (
            invocation.assignment.action_type
            is ResearchActionType.PLAN_NEXT_ACTION
        ):
            return self._planner.perform(invocation)
        return self._scientist.perform(invocation)
