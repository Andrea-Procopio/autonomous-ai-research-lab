"""The research director: wiring for one orchestration decision.

The director owns no scientific judgment of its own. It composes the three
functions the decision splits into —

    CandidateGenerator   what could we do?
    UtilityEvaluator     how valuable might each option be?
    SearchPolicy         what do we take, given uncertainty and resources?

— and preserves the full decision as a
:class:`~autonomous_research_lab.core.decision.DecisionRecord`: state before,
every candidate with its utility, who generated and who evaluated, what was
selected. The caller completes the record once the attempt's outcome is known.

Any of the three parts swaps independently: a model-backed generator, a learned
evaluator, or a bandit policy each slot in without touching the other two.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.actions import ResearchAction
from ..core.decision import DecisionRecord, EvaluatedCandidate
from ..core.state import ResearchState
from ..search.policy import SearchPolicy
from .candidates import CandidateGenerator
from .evaluation import UtilityEvaluator


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision: the selected candidate (if any) and its full record."""

    record: DecisionRecord
    selected: EvaluatedCandidate | None

    @property
    def action(self) -> ResearchAction | None:
        return self.selected.action if self.selected else None


class ResearchDirector:
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
