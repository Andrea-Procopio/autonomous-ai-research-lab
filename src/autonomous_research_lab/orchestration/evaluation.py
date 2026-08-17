"""Utility evaluation: how scientifically valuable might each candidate be.

An evaluator maps ``(state, candidate)`` to a multi-dimensional
:class:`~autonomous_research_lab.core.decision.ActionUtility`. It estimates;
it does not choose. Estimates name their method so that estimate quality can
itself be studied from trajectory logs later — the interface admits heuristic,
model-judged, and learned evaluators equally.

``HeuristicUtilityEvaluator`` is a deterministic stub: fixed per-action-type
profiles, honest about being guesses (every profile carries maximal
``estimate_uncertainty``). It exists so the loop runs and the decision record
has real structure, not because these numbers mean anything.
"""

from __future__ import annotations

from typing import Final, Protocol

from ..core.actions import ResearchActionType
from ..core.budget import ResourceCost
from ..core.decision import ActionCandidate, ActionUtility
from ..core.state import ResearchState


class UtilityEvaluator(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(
        self, state: ResearchState, candidate: ActionCandidate
    ) -> ActionUtility: ...


_METHOD: Final = "heuristic:v0"

_PROFILES: Final[dict[ResearchActionType, ActionUtility]] = {
    ResearchActionType.GENERATE_HYPOTHESIS: ActionUtility(
        expected_information_gain=1.0,
        novelty=0.5,
        expected_success_probability=0.9,
        expected_cost=ResourceCost(wall_clock_seconds=30.0, model_tokens=4_000),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.DERIVE_PREDICTION: ActionUtility(
        expected_information_gain=1.0,
        discrimination_value=1.0,
        expected_success_probability=0.9,
        expected_cost=ResourceCost(wall_clock_seconds=20.0, model_tokens=3_000),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.DESIGN_EXPERIMENT: ActionUtility(
        expected_information_gain=1.5,
        discrimination_value=1.0,
        expected_success_probability=0.85,
        expected_cost=ResourceCost(wall_clock_seconds=60.0, model_tokens=8_000),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.RUN_EXPERIMENT: ActionUtility(
        expected_information_gain=2.0,
        discrimination_value=2.0,
        expected_success_probability=0.8,
        expected_cost=ResourceCost(wall_clock_seconds=300.0),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.ANALYZE: ActionUtility(
        expected_information_gain=1.5,
        expected_success_probability=0.9,
        expected_cost=ResourceCost(wall_clock_seconds=30.0, model_tokens=4_000),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.SYNTHESIZE_FINDING: ActionUtility(
        expected_information_gain=1.0,
        importance=0.5,
        expected_success_probability=0.9,
        expected_cost=ResourceCost(wall_clock_seconds=30.0, model_tokens=6_000),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.ASSESS_CLAIM: ActionUtility(
        expected_information_gain=1.0,
        importance=1.0,
        expected_success_probability=0.9,
        expected_cost=ResourceCost(wall_clock_seconds=20.0, model_tokens=4_000),
        estimate_uncertainty=1.0,
        method=_METHOD,
    ),
    ResearchActionType.STOP_INVESTIGATION: ActionUtility(
        # Explicitly zero, not unestimated: stopping yields no information,
        # and it must never outrank open work on a default.
        expected_information_gain=0.0,
        expected_success_probability=1.0,
        estimate_uncertainty=0.0,
        method=_METHOD,
    ),
}

_FALLBACK: Final = ActionUtility(estimate_uncertainty=1.0, method=_METHOD)


class HeuristicUtilityEvaluator:
    """Fixed profiles per action type. A placeholder with a name, so its
    estimates remain attributable — and dismissible — in trajectory logs."""

    @property
    def name(self) -> str:
        return _METHOD

    def evaluate(
        self,
        state: ResearchState,  # noqa: ARG002 - state-blind stub; part of the contract
        candidate: ActionCandidate,
    ) -> ActionUtility:
        return _PROFILES.get(candidate.action.action_type, _FALLBACK)
