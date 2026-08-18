"""The deterministic governance seat over a model-backed planner.

:class:`PlanningDirector` implements the existing
:class:`~autonomous_research_lab.orchestration.director.FrontierDirector`
protocol, so the unmodified :class:`~.loop.ResearchRuntime` remains the one
orchestration loop. Its policy is a fixed priority, no model calls:

1. an open (undispatched) **stop** decision becomes ``STOP_INVESTIGATION``
   with the typed reason in its rationale — the loop's existing halt path;
2. a **pending experiment** becomes ``RUN_EXPERIMENT`` (this is how an
   accepted new-experiment or ablation decision reaches the engineer);
3. an open **replicate** decision whose target still has a replication gap
   becomes ``REPLICATE``;
4. otherwise the planner itself is invoked: ``PLAN_NEXT_ACTION``.

The planner proposes; this seat decides what is put in front of the
governed commit next. Each decision is marked *dispatched* — durably,
write-once — when its follow-up action is emitted, so a decision is acted
on exactly once even across restarts. A decision whose follow-up is no
longer actionable (its chain never committed, or its gap closed) is marked
dispatched as stale rather than silently forgotten.

Budget exhaustion needs no rule here: the runtime's own budget gate halts
the program when the selected action's cost is unaffordable, which is the
"stop naturally under an explicit budget" path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.actions import ResearchAction, ResearchActionType
from ..core.budget import NO_COST, ResourceCost
from ..core.decision import ActionCandidate
from ..core.experiment import ExperimentSpec
from ..runtime.escalation import CompactValuation, Level, ReasoningTier
from ..runtime.frontier import ResearchFrontier
from ..runtime.planning_store import PlanningAction, PlanningStore
from ..runtime.playbook import PlaybookAdvice
from .director import Deliberation, RuleBasedFrontierDirector, ValuedCandidate
from .synthesis import SynthesisReview

#: What one planning invocation is estimated to cost — model thinking, no
#: execution. Deliberately coarse, like the rule-based director's costs.
DEFAULT_PLAN_COST = ResourceCost(wall_clock_seconds=120.0, model_tokens=24_000)

_RUN_COST_FLOOR = ResourceCost(wall_clock_seconds=300.0)


@dataclass
class PlanningDirector:
    """See the module docstring. ``synthesize`` delegates to the rule-based
    director: the slow loop's summary is not planning work."""

    plans: PlanningStore
    plan_cost: ResourceCost = DEFAULT_PLAN_COST
    _fallback: RuleBasedFrontierDirector = field(
        default_factory=RuleBasedFrontierDirector
    )

    @property
    def name(self) -> str:
        return "frontier-director:planning:v1"

    def deliberate(
        self,
        frontier: ResearchFrontier,
        *,
        advice: PlaybookAdvice | None = None,  # noqa: ARG002 - fixed policy
        tier: ReasoningTier = ReasoningTier.ROUTINE,
        max_candidates: int = 3,  # noqa: ARG002 - one candidate per step
    ) -> Deliberation:
        candidate = self._next_candidate(frontier)
        return Deliberation(
            candidates=(candidate,),
            selected_action_id=candidate.action.id,
            reasoning=candidate.valuation.rationale,
            tier=tier,
        )

    def synthesize(
        self,
        frontier: ResearchFrontier,
        *,
        tier: ReasoningTier = ReasoningTier.STRONG,
    ) -> SynthesisReview:
        return self._fallback.synthesize(frontier, tier=tier)

    # -- the fixed priority ----------------------------------------------------

    def _next_candidate(self, frontier: ResearchFrontier) -> ValuedCandidate:
        open_decisions = self.plans.open_decisions()

        for record in open_decisions:
            if record.action is PlanningAction.STOP:
                reason = (
                    record.stop_reason.value
                    if record.stop_reason is not None
                    else "unspecified"
                )
                self.plans.mark_dispatched(
                    record.id, "stop dispatched to the halt path"
                )
                return self._candidate(
                    ResearchActionType.STOP_INVESTIGATION,
                    rationale=(
                        f"planner stop: {reason} — {record.rationale} "
                        f"(decision {record.id})"
                    ),
                    targets=(),
                    cost=NO_COST,
                )

        if frontier.pending_experiments:
            spec = frontier.pending_experiments[0]
            note = ""
            for record in open_decisions:
                if record.spec_id == spec.id and record.action in {
                    PlanningAction.NEW_EXPERIMENT,
                    PlanningAction.ABLATION,
                }:
                    self.plans.mark_dispatched(
                        record.id, f"run_experiment emitted for {spec.id}"
                    )
                    note = f" (planner decision {record.id})"
                    break
            return self._candidate(
                ResearchActionType.RUN_EXPERIMENT,
                rationale=f"experiment {spec.id} is designed but has no "
                f"result{note}",
                targets=(spec.id,),
                cost=_run_cost(spec),
            )

        gaps = {spec.id: spec for spec in frontier.replication_gaps}
        for record in open_decisions:
            if record.action is PlanningAction.REPLICATE:
                target = gaps.get(record.spec_id)
                if target is None:
                    self.plans.mark_dispatched(
                        record.id,
                        "stale: the target no longer has a replication gap",
                    )
                    continue
                self.plans.mark_dispatched(
                    record.id, f"replicate emitted for {target.id}"
                )
                return self._candidate(
                    ResearchActionType.REPLICATE,
                    rationale=(
                        f"planner decision {record.id}: replicate "
                        f"{target.id} at seed {record.replication_seed!r}"
                    ),
                    targets=(target.id,),
                    cost=_run_cost(target),
                )

        for record in open_decisions:
            if record.action in {
                PlanningAction.NEW_EXPERIMENT,
                PlanningAction.ABLATION,
            }:
                # Its chain never reached the state (a later gate refused
                # the bundle): stale, recorded as such, planned over.
                self.plans.mark_dispatched(
                    record.id, "stale: the decision's chain never committed"
                )

        return self._candidate(
            ResearchActionType.PLAN_NEXT_ACTION,
            rationale=(
                "no pending execution work and no open planning decision: "
                "the planner selects the next scientific action"
            ),
            targets=(),
            cost=self.plan_cost,
        )

    def _candidate(
        self,
        action_type: ResearchActionType,
        *,
        rationale: str,
        targets: tuple[str, ...],
        cost: ResourceCost,
    ) -> ValuedCandidate:
        return ValuedCandidate(
            candidate=ActionCandidate(
                action=ResearchAction(
                    action_type=action_type,
                    rationale=rationale,
                    targets=targets,
                ),
                generated_by=self.name,
            ),
            valuation=CompactValuation(
                scientific_value=Level.HIGH,
                expected_cost=cost,
                uncertainty=Level.MEDIUM,
                rationale=rationale,
            ),
        )


def _run_cost(spec: ExperimentSpec) -> ResourceCost:
    return (
        spec.estimated_cost
        if not spec.estimated_cost.is_zero
        else _RUN_COST_FLOOR
    )
