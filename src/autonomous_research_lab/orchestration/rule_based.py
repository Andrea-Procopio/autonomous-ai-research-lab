"""A rule-based reference director.

This exists to exercise the contracts end to end without a model provider. It
closes obvious gaps in the state -- a hypothesis with no experiment, an
experiment with no result, a result nobody has read -- and stops when there are
none. That is bookkeeping, not scientific judgement, and it is not a stand-in
for the real director.

It reads which work is already done from ``state.history`` rather than from the
evidence store, so the director stays a pure function of the state. The history
is the record of what was actually done, which is the same reason it is stored
at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.actions import ResearchAction, ResearchActionType
from ..core.budget import ResourceCost
from ..core.state import ResearchState
from .director import ResearchDirector

_COSTS = {
    ResearchActionType.GENERATE_HYPOTHESIS: ResourceCost(
        wall_clock_seconds=30.0, model_tokens=4_000
    ),
    ResearchActionType.DESIGN_EXPERIMENT: ResourceCost(
        wall_clock_seconds=60.0, model_tokens=8_000
    ),
    ResearchActionType.RUN_EXPERIMENT: ResourceCost(wall_clock_seconds=300.0),
    ResearchActionType.ANALYZE: ResourceCost(
        wall_clock_seconds=30.0, model_tokens=4_000
    ),
    ResearchActionType.SYNTHESIZE_FINDING: ResourceCost(
        wall_clock_seconds=30.0, model_tokens=6_000
    ),
}


class RuleBasedDirector(ResearchDirector):
    def candidate_actions(self, state: ResearchState) -> Sequence[ResearchAction]:
        candidates: list[ResearchAction] = []

        if not state.hypotheses:
            candidates.append(
                _action(
                    ResearchActionType.GENERATE_HYPOTHESIS,
                    "no hypothesis exists for the stated objective",
                    tuple(q.id for q in state.questions),
                )
            )

        for hypothesis in state.hypotheses:
            if hypothesis.status.is_terminal:
                continue
            if not state.experiments_for(hypothesis.id):
                candidates.append(
                    _action(
                        ResearchActionType.DESIGN_EXPERIMENT,
                        f"hypothesis {hypothesis.id} has no experiment testing it",
                        (hypothesis.id,),
                    )
                )

        for spec in state.experiments:
            if not state.results_for(spec.id):
                candidates.append(
                    _action(
                        ResearchActionType.RUN_EXPERIMENT,
                        f"experiment {spec.id} has been designed but not run",
                        (spec.id,),
                    )
                )

        analyzed = _targets(state, ResearchActionType.ANALYZE)
        for ref in state.results:
            if ref.result_id not in analyzed:
                candidates.append(
                    _action(
                        ResearchActionType.ANALYZE,
                        f"result {ref.result_id} has not been read into evidence",
                        (ref.result_id,),
                    )
                )

        synthesized = _targets(state, ResearchActionType.SYNTHESIZE_FINDING)
        for evidence_id in state.evidence_ids:
            if evidence_id not in synthesized:
                candidates.append(
                    _action(
                        ResearchActionType.SYNTHESIZE_FINDING,
                        f"evidence {evidence_id} has not been turned into a claim",
                        (evidence_id,),
                    )
                )

        if candidates:
            return candidates
        # Offered only when nothing else is available. A zero-cost action is
        # unbeatable under an information-per-cost score, so making stopping a
        # standing candidate would make the system stop immediately, always.
        return [
            _action(
                ResearchActionType.STOP_INVESTIGATION,
                "no open scientific work remains in this state",
                (),
                gain=0.0,
            )
        ]


def _action(
    action_type: ResearchActionType,
    rationale: str,
    targets: tuple[str, ...],
    *,
    gain: float | None = None,
) -> ResearchAction:
    return ResearchAction(
        action_type=action_type,
        rationale=rationale,
        targets=targets,
        estimated_cost=_COSTS.get(action_type, ResourceCost()),
        expected_information_gain=gain,
    )


def _targets(state: ResearchState, action_type: ResearchActionType) -> frozenset[str]:
    return frozenset(
        target
        for action in state.history
        if action.action_type is action_type
        for target in action.targets
    )
