"""The research director: one reasoning seat, two ways to wire it.

**The fast path** — :class:`FrontierDirector` — is the runtime default. One
invocation receives a :class:`~autonomous_research_lab.runtime.frontier.
ResearchFrontier` (never the raw state) and performs candidate generation,
coarse valuation and selection *in the same deliberation*, returning a
:class:`Deliberation`. When a real model sits behind this protocol, an
ordinary decision costs exactly one call. The intermediate candidate set and
its valuations are still logged — :func:`deliberation_record` converts a
deliberation into the same :class:`~autonomous_research_lab.core.decision.
DecisionRecord` the trajectory has always used, with the director named as
generator, evaluator *and* policy, which is precisely what happened.

**The decomposed path** — :class:`ResearchDirector` below — keeps the three
separated functions (generator → evaluator → policy) available. It costs up
to three reasoning invocations when those become model-backed, and exists for
the ablation: whether the separation buys decision quality is a question the
trajectories should answer, not an assumption the runtime should pay for on
every step.

The director also owns the slow loop: :meth:`FrontierDirector.synthesize` is
the same seat in a stronger reasoning mode, not a second agent.

:class:`RuleBasedFrontierDirector` is the deterministic reference
implementation: it closes structural gaps on the frontier in a fixed
priority order, values candidates on the coarse HIGH/MEDIUM/LOW scale, and
prefers pairwise "A over B because ..." rationale to absolute scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.actions import ResearchAction, ResearchActionType
from ..core.budget import NO_COST, ResourceCost
from ..core.decision import ActionCandidate, DecisionRecord, EvaluatedCandidate
from ..core.state import ResearchState
from ..runtime.escalation import (
    CompactValuation,
    Level,
    ReasoningTier,
    as_action_utility,
)
from ..runtime.frontier import ResearchFrontier
from ..runtime.playbook import PlaybookAdvice
from ..search.policy import SearchPolicy
from .candidates import CandidateGenerator
from .evaluation import UtilityEvaluator
from .synthesis import SynthesisRecommendation, SynthesisReview

# -- the fast path -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValuedCandidate:
    """A candidate action with the director's coarse valuation of it."""

    candidate: ActionCandidate
    valuation: CompactValuation

    @property
    def action(self) -> ResearchAction:
        return self.candidate.action


@dataclass(frozen=True, slots=True)
class Deliberation:
    """The complete output of one director invocation: the candidate set,
    the valuations, the choice, and the reasoning — one call, all three
    jobs, everything preserved."""

    candidates: tuple[ValuedCandidate, ...]
    selected_action_id: str | None
    reasoning: str
    tier: ReasoningTier

    @property
    def selected(self) -> ValuedCandidate | None:
        return next(
            (c for c in self.candidates if c.action.id == self.selected_action_id),
            None,
        )


class FrontierDirector(Protocol):
    """One reasoning seat over the frontier. ``deliberate`` is the fast
    loop; ``synthesize`` is the same seat in slow-loop mode."""

    @property
    def name(self) -> str: ...

    def deliberate(
        self,
        frontier: ResearchFrontier,
        *,
        advice: PlaybookAdvice | None = None,
        tier: ReasoningTier = ReasoningTier.ROUTINE,
        max_candidates: int = 3,
    ) -> Deliberation: ...

    def synthesize(
        self,
        frontier: ResearchFrontier,
        *,
        tier: ReasoningTier = ReasoningTier.STRONG,
    ) -> SynthesisReview: ...


def deliberation_record(
    deliberation: Deliberation, *, state_id: str, director: str
) -> DecisionRecord:
    """Preserve a fast-path deliberation as a standard decision record.

    The compact valuations are embedded into ``ActionUtility`` (ordinal
    embedding, named as such), and the director is recorded as generator,
    evaluator and policy at once — the trajectory says truthfully that one
    invocation did all three jobs.
    """
    return DecisionRecord(
        state_before_id=state_id,
        evaluated=tuple(
            EvaluatedCandidate(
                candidate=c.candidate, utility=as_action_utility(c.valuation)
            )
            for c in deliberation.candidates
        ),
        selected_action_id=deliberation.selected_action_id,
        generator=director,
        evaluator=director,
        policy=director,
    )


_THINK_COST = ResourceCost(wall_clock_seconds=30.0, model_tokens=4_000)
_RUN_COST = ResourceCost(wall_clock_seconds=300.0)


class RuleBasedFrontierDirector:
    """Deterministic reference director: directed refinement, no branching.

    Priority follows the scientific chain — an untestable hypothesis is
    worth more work than an unassessed claim — and replication is boosted
    when the playbook recommends it. Stopping is offered only when the
    frontier has no open work, for the standing reason: a free stop would
    dominate any ranking.
    """

    @property
    def name(self) -> str:
        return "frontier-director:rule-based:v1"

    def deliberate(
        self,
        frontier: ResearchFrontier,
        *,
        advice: PlaybookAdvice | None = None,
        tier: ReasoningTier = ReasoningTier.ROUTINE,
        max_candidates: int = 3,
    ) -> Deliberation:
        candidates = self._candidates(frontier, advice, max_candidates)
        selected = max(
            candidates,
            key=lambda c: (
                c.valuation.scientific_value.ordinal,
                -_scalar_cost(c.valuation.expected_cost),
                c.action.id,
            ),
        )
        return Deliberation(
            candidates=candidates,
            selected_action_id=selected.action.id,
            reasoning=self._reasoning(candidates, selected, advice),
            tier=tier,
        )

    def synthesize(
        self,
        frontier: ResearchFrontier,
        *,
        tier: ReasoningTier = ReasoningTier.STRONG,
    ) -> SynthesisReview:
        open_work = (
            frontier.hypotheses_without_predictions
            or frontier.untested_predictions
            or frontier.pending_experiments
            or frontier.replication_gaps
            or frontier.unsynthesized_evidence
            or frontier.unassessed_claims
        )
        if frontier.contradictions:
            recommendation = SynthesisRecommendation.REPLICATE
        elif not open_work and not frontier.active_hypotheses:
            recommendation = SynthesisRecommendation.STOP
        else:
            recommendation = SynthesisRecommendation.CONTINUE
        summary = (
            f"{len(frontier.active_hypotheses)} active and "
            f"{len(frontier.settled_hypotheses)} settled hypothesis(es); "
            f"{len(frontier.best_findings)} settled finding(s); "
            f"{len(frontier.contradictions)} contradiction(s) on the record; "
            f"{len(frontier.failed_attempts)} unresolved failure(s). "
            f"Recommendation: {recommendation}."
        )
        return SynthesisReview(
            summary=summary, recommendation=recommendation, tier=tier
        )

    # -- candidate construction ---------------------------------------------

    def _candidates(
        self,
        frontier: ResearchFrontier,
        advice: PlaybookAdvice | None,
        max_candidates: int,
    ) -> tuple[ValuedCandidate, ...]:
        offers: list[ValuedCandidate] = []

        def offer(
            action_type: ResearchActionType,
            rationale: str,
            targets: tuple[str, ...],
            value: Level,
            cost: ResourceCost,
            uncertainty: Level = Level.MEDIUM,
        ) -> None:
            if len(offers) >= max_candidates:
                return
            offers.append(
                ValuedCandidate(
                    candidate=ActionCandidate(
                        action=ResearchAction(
                            action_type=action_type,
                            rationale=rationale,
                            targets=targets,
                        ),
                        generated_by=self.name,
                    ),
                    valuation=CompactValuation(
                        scientific_value=value,
                        expected_cost=cost,
                        uncertainty=uncertainty,
                        rationale=rationale,
                    ),
                )
            )

        if (
            frontier.open_questions
            and not frontier.active_hypotheses
            and not frontier.settled_hypotheses
        ):
            offer(
                ResearchActionType.GENERATE_HYPOTHESIS,
                "the open question has no hypothesis answering it",
                tuple(q.id for q in frontier.open_questions),
                Level.HIGH,
                _THINK_COST,
                uncertainty=Level.HIGH,
            )
        for hypothesis in frontier.hypotheses_without_predictions:
            offer(
                ResearchActionType.DERIVE_PREDICTION,
                f"hypothesis {hypothesis.id} has no testable prediction",
                (hypothesis.id,),
                Level.HIGH,
                _THINK_COST,
            )
        for prediction in frontier.untested_predictions:
            offer(
                ResearchActionType.DESIGN_EXPERIMENT,
                f"prediction {prediction.id} has no experiment testing it",
                (prediction.id,),
                Level.HIGH,
                _THINK_COST,
            )
        for spec in frontier.pending_experiments:
            offer(
                ResearchActionType.RUN_EXPERIMENT,
                f"experiment {spec.id} is designed but has never run",
                (spec.id,),
                Level.HIGH,
                spec.estimated_cost if not spec.estimated_cost.is_zero else _RUN_COST,
            )
        replicate_urged = advice is not None and advice.stage.name in {
            "replicate_promising_result",
            "investigate_mechanism",
        }
        for spec in frontier.replication_gaps:
            rationale = f"experiment {spec.id} has unused declared seeds"
            if replicate_urged and advice is not None:
                rationale += f"; playbook urges it: {advice.rationale}"
            offer(
                ResearchActionType.REPLICATE,
                rationale,
                (spec.id,),
                Level.HIGH if replicate_urged else Level.MEDIUM,
                spec.estimated_cost if not spec.estimated_cost.is_zero else _RUN_COST,
            )
        for evidence_id in frontier.unsynthesized_evidence:
            offer(
                ResearchActionType.SYNTHESIZE_FINDING,
                f"evidence {evidence_id} has not been turned into a claim",
                (evidence_id,),
                Level.MEDIUM,
                _THINK_COST,
            )
        for claim in frontier.unassessed_claims:
            offer(
                ResearchActionType.ASSESS_CLAIM,
                f"claim {claim.id} has never been epistemically assessed",
                (claim.id,),
                Level.MEDIUM,
                _THINK_COST,
            )

        if not offers:
            offer(
                ResearchActionType.STOP_INVESTIGATION,
                "no open scientific work remains on the frontier",
                (),
                Level.LOW,
                NO_COST,
                uncertainty=Level.LOW,
            )
        return tuple(offers)

    def _reasoning(
        self,
        candidates: tuple[ValuedCandidate, ...],
        selected: ValuedCandidate,
        advice: PlaybookAdvice | None,
    ) -> str:
        lines = []
        if advice is not None:
            lines.append(
                f"playbook suggests {advice.stage.name} ({advice.rationale})"
            )
        for other in candidates:
            if other.action.id == selected.action.id:
                continue
            # Pairwise, not absolute: the comparison names its ground.
            if (
                selected.valuation.scientific_value.ordinal
                > other.valuation.scientific_value.ordinal
            ):
                ground = (
                    f"{selected.valuation.scientific_value} value beats "
                    f"{other.valuation.scientific_value}"
                )
            else:
                ground = "equal value, lower or equal expected cost"
            lines.append(
                f"prefer {selected.action.action_type} over "
                f"{other.action.action_type}: {ground}"
            )
        lines.append(
            f"selected {selected.action.action_type}: "
            f"{selected.valuation.rationale}"
        )
        return "; ".join(lines)


def _scalar_cost(cost: ResourceCost) -> float:
    """The director's own crude exchange rate, used only to break ties
    between equally valued candidates."""
    return (
        cost.usd
        + cost.gpu_hours
        + cost.wall_clock_seconds / 3600.0
        + cost.model_tokens / 1_000_000.0
    )


# -- the decomposed path (kept for the ablation) -----------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision: the selected candidate (if any) and its full record."""

    record: DecisionRecord
    selected: EvaluatedCandidate | None

    @property
    def action(self) -> ResearchAction | None:
        return self.selected.action if self.selected else None


class ResearchDirector:
    """The decomposed wiring: generator → evaluator → policy, each
    separately swappable — and each separately billable once model-backed.
    The runtime does not require this path; it exists so the value of the
    separation can be measured rather than assumed."""

    def __init__(
        self,
        generator: CandidateGenerator,
        evaluator: UtilityEvaluator,
        policy: SearchPolicy,
    ) -> None:
        self._generator = generator
        self._evaluator = evaluator
        self._policy = policy

    def decide(self, state: ResearchState) -> Decision:
        candidates = self._generator.generate(state)
        evaluated = tuple(
            EvaluatedCandidate(
                candidate=candidate,
                utility=self._evaluator.evaluate(state, candidate),
            )
            for candidate in candidates
        )
        selected = self._policy.select(state, evaluated)
        record = DecisionRecord(
            state_before_id=state.id,
            evaluated=evaluated,
            selected_action_id=selected.action.id if selected else None,
            generator=self._generator.name,
            evaluator=self._evaluator.name,
            policy=self._policy.name,
        )
        return Decision(record=record, selected=selected)
