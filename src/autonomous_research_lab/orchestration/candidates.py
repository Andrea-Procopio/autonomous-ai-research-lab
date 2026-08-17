"""Candidate generation: what could be done next.

A generator enumerates scientifically available actions. It does not estimate
their value (utility evaluation) and does not choose among them (search
policy). The three are separate functions with separate failure modes.

``RuleBasedCandidateGenerator`` is the reference implementation: it closes
structural gaps in the state and stops when there are none. Completion is read
from **facts and succeeded attempts**, never from ``history``:

* fact-based where a fact exists — an experiment is run when a result for its
  spec is recorded;
* attempt-based where the product is a judgment — a result is analyzed when an
  ANALYZE attempt targeting it *succeeded*. A failed attempt leaves the work
  open, so it is re-offered and a retry is a new attempt.

Work with an attempt currently queued or running is not re-offered.

Propositions carry no status, so the generator *consumes* epistemic judgments
where it needs standing: a hypothesis whose current assessment is SUPPORTED or
REFUTED is treated as settled and not offered further derivation work. That is
a generation policy reading assessments — the judgment itself was made
elsewhere, by an assessor who signed it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol

from ..core.actions import ResearchAction, ResearchActionType
from ..core.assessment import AssessmentVerdict
from ..core.decision import ActionCandidate
from ..core.hypothesis import Hypothesis
from ..core.state import ResearchState

#: Verdicts this generator treats as settling a hypothesis. A generation
#: policy, not epistemology: the assessments were made elsewhere.
_SETTLED: Final = frozenset(
    {AssessmentVerdict.SUPPORTED, AssessmentVerdict.REFUTED}
)


class CandidateGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, state: ResearchState) -> Sequence[ActionCandidate]: ...


class RuleBasedCandidateGenerator:
    @property
    def name(self) -> str:
        return "rule-based:v1"

    def generate(self, state: ResearchState) -> Sequence[ActionCandidate]:
        candidates: list[ActionCandidate] = []

        def offer(
            action_type: ResearchActionType, rationale: str, *targets: str
        ) -> None:
            if state.in_flight(action_type, targets[0] if targets else None):
                return
            candidates.append(
                ActionCandidate(
                    action=ResearchAction(
                        action_type=action_type,
                        rationale=rationale,
                        targets=targets,
                    ),
                    generated_by=self.name,
                )
            )

        def settled(hypothesis: Hypothesis) -> bool:
            assessment = state.current_assessment(hypothesis.id)
            return assessment is not None and assessment.verdict in _SETTLED

        if not state.hypotheses:
            offer(
                ResearchActionType.GENERATE_HYPOTHESIS,
                "no hypothesis exists for the stated objective",
                *(q.id for q in state.questions),
            )

        for hypothesis in state.hypotheses:
            if settled(hypothesis):
                continue
            if not state.predictions_for(hypothesis.id):
                offer(
                    ResearchActionType.DERIVE_PREDICTION,
                    f"hypothesis {hypothesis.id} has no testable prediction",
                    hypothesis.id,
                )

        for prediction in state.predictions:
            owner = state.hypothesis(prediction.hypothesis_id)
            if owner is not None and settled(owner):
                continue
            if not state.experiments_for(prediction.id):
                offer(
                    ResearchActionType.DESIGN_EXPERIMENT,
                    f"prediction {prediction.id} has no experiment testing it",
                    prediction.id,
                )

        for spec in state.experiments:
            if not state.results_for(spec.id):
                offer(
                    ResearchActionType.RUN_EXPERIMENT,
                    f"experiment {spec.id} has been designed but not run",
                    spec.id,
                )

        for ref in state.results:
            if not state.has_succeeded(ResearchActionType.ANALYZE, ref.result_id):
                offer(
                    ResearchActionType.ANALYZE,
                    f"result {ref.result_id} has not been read into evidence",
                    ref.result_id,
                )

        for evidence_id in state.evidence_ids:
            if not state.has_succeeded(
                ResearchActionType.SYNTHESIZE_FINDING, evidence_id
            ):
                offer(
                    ResearchActionType.SYNTHESIZE_FINDING,
                    f"evidence {evidence_id} has not been turned into a claim",
                    evidence_id,
                )

        for claim in state.claims:
            if state.current_assessment(claim.id) is None:
                offer(
                    ResearchActionType.ASSESS_CLAIM,
                    f"claim {claim.id} has never been epistemically assessed",
                    claim.id,
                )

        if candidates:
            return candidates
        # Offered only when nothing else is available: a free stop action would
        # dominate any value-per-cost ranking if it were a standing candidate.
        return [
            ActionCandidate(
                action=ResearchAction(
                    action_type=ResearchActionType.STOP_INVESTIGATION,
                    rationale="no open scientific work remains in this state",
                ),
                generated_by=self.name,
            )
        ]
