"""The composite seat: who speaks when, pinned rule by rule."""

from __future__ import annotations

import json
from pathlib import Path

from autonomous_research_lab.core.actions import ResearchActionType
from autonomous_research_lab.core.assessment import (
    AssessmentVerdict,
    EpistemicAssessment,
)
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    ExperimentSpec,
    ExperimentStatus,
    ResultRef,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.orchestration.planning import PlanningDirector
from autonomous_research_lab.roles.base import RoleContext
from autonomous_research_lab.roles.planner import (
    PLANNING_SCHEMA,
    check_decision,
    render_planning_context,
)
from autonomous_research_lab.runtime.frontier import ResearchFrontier
from autonomous_research_lab.runtime.planning_store import PlanningStore
from autonomous_research_lab.runtime.providers import (
    Message,
    MessageRole,
    ModelRequest,
)
from examples.vision_lab.catalog import catalog_for
from examples.vision_lab.direction import VisionDirector
from examples.vision_lab.scripted import _planning_decision

ENCODER_METRIC = (
    "difference in linear probe accuracy: "
    "trained encoder minus randomly initialized encoder"
)

QUESTION = ResearchQuestion(text="What does training explain?")
HYPOTHESIS = Hypothesis(
    statement="Training helps.", question_id=QUESTION.id
)
PREDICTION = Prediction(
    hypothesis_id=HYPOTHESIS.id,
    condition="on held-out images",
    metric=ENCODER_METRIC,
    comparator=Comparator.GREATER_THAN,
    threshold=0.0,
    expectation="the contrast is positive",
)
SPEC = ExperimentSpec(
    prediction_id=PREDICTION.id,
    objective="measure the contrast",
    procedure="run the template",
    metrics=(ENCODER_METRIC,),
    seeds=(11,),
)
EVIDENCE = Evidence(
    result_id="res_1",
    spec_id=SPEC.id,
    kind=EvidenceKind.MEASUREMENT,
    observation="contrast 0.05 at seed 11",
)
FINDING = EpistemicAssessment(
    subject_id="clm_1",
    verdict=AssessmentVerdict.SUPPORTED,
    method="test",
    evidence_ids=(EVIDENCE.id,),
)


def frontier(**overrides: object) -> ResearchFrontier:
    defaults: dict[str, object] = {
        "state_id": "st_1",
        "objective": "o",
        "open_questions": (),
        "active_hypotheses": (),
        "settled_hypotheses": (),
        "hypotheses_without_predictions": (),
        "untested_predictions": (),
        "unresolved_predictions": (),
        "pending_experiments": (),
        "replication_gaps": (),
        "recent_results": (),
        "unsynthesized_evidence": (),
        "unassessed_claims": (),
        "contradictions": (),
        "failed_attempts": (),
        "best_findings": (),
        "open_decisions": (),
        "remaining_budget": ResearchBudget(
            wall_clock_seconds=3600.0, model_tokens=100_000
        ),
    }
    return ResearchFrontier(**(defaults | overrides))  # type: ignore[arg-type]


class Spy(PlanningDirector):
    """Counts deliberations, so the one-call invariant is checkable."""

    def __init__(self, plans: PlanningStore) -> None:
        super().__init__(plans=plans)
        self.calls = 0

    def deliberate(self, frontier, *, advice=None, tier=None, max_candidates=3):  # type: ignore[no-untyped-def]
        self.calls += 1
        from autonomous_research_lab.runtime.escalation import ReasoningTier

        return super().deliberate(
            frontier,
            advice=advice,
            tier=tier or ReasoningTier.ROUTINE,
            max_candidates=max_candidates,
        )


def director(tmp_path: Path) -> tuple[VisionDirector, Spy, PlanningStore]:
    plans = PlanningStore(tmp_path / "planning")
    spy = Spy(plans)
    return VisionDirector(plans=plans, planning=spy), spy, plans


class TestRouting:
    def test_structural_work_never_reaches_the_planner(
        self, tmp_path: Path
    ) -> None:
        composite, spy, _ = director(tmp_path)

        chosen = composite.deliberate(
            frontier(untested_predictions=(PREDICTION,))
        )

        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.DESIGN_EXPERIMENT
        )
        assert spy.calls == 0

    def test_a_bootstrap_run_keeps_rule_based_economics(
        self, tmp_path: Path
    ) -> None:
        composite, spy, _ = director(tmp_path)

        chosen = composite.deliberate(frontier(pending_experiments=(SPEC,)))

        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.RUN_EXPERIMENT
        )
        assert spy.calls == 0

    def test_a_bare_replication_gap_never_reaches_the_planner(
        self, tmp_path: Path
    ) -> None:
        """A planner-less gap delegated to the planning director would
        fall through to an unintended, billed consultation."""
        composite, spy, _ = director(tmp_path)

        chosen = composite.deliberate(frontier(replication_gaps=(SPEC,)))

        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type is ResearchActionType.REPLICATE
        )
        assert spy.calls == 0

    def test_a_stop_without_findings_stays_a_stop(
        self, tmp_path: Path
    ) -> None:
        composite, spy, _ = director(tmp_path)

        chosen = composite.deliberate(frontier())

        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.STOP_INVESTIGATION
        )
        assert "no open scientific work" in chosen.selected.action.rationale
        assert spy.calls == 0

    def test_a_stop_with_findings_becomes_a_consultation(
        self, tmp_path: Path
    ) -> None:
        composite, spy, _ = director(tmp_path)

        chosen = composite.deliberate(frontier(best_findings=(FINDING,)))

        assert spy.calls == 1
        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.PLAN_NEXT_ACTION
        )

    def test_the_planning_director_is_called_at_most_once(
        self, tmp_path: Path
    ) -> None:
        composite, spy, _ = director(tmp_path)
        composite.deliberate(frontier(best_findings=(FINDING,)))
        composite.deliberate(frontier(best_findings=(FINDING,)))
        assert spy.calls == 2  # once per step, never more

    def test_a_terminally_rejected_consultation_ends_delegation(
        self, tmp_path: Path
    ) -> None:
        """The durable guard: a rejected invocation that never became a
        record silences the planner — the frontier's failed-attempt view
        goes blind to it after any successful consultation."""
        composite, spy, plans = director(tmp_path)
        plans.preserve_rejected(
            invocation_id="inv_dead",
            reasons=(("unknown_question", "no question cited"),),
            request_fingerprint="f",
            response_id="r",
            payload={"action": "stop"},
            repair=1,
        )

        chosen = composite.deliberate(frontier(best_findings=(FINDING,)))

        assert spy.calls == 0
        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.STOP_INVESTIGATION
        )


class TestScriptedDecisions:
    """The scripted payloads, held to the real gate over a really
    rendered context — the same text the live prompt would carry."""

    def context(self) -> RoleContext:
        return RoleContext(
            questions=(QUESTION,),
            hypotheses=(HYPOTHESIS,),
            predictions=(PREDICTION,),
            experiments=(SPEC,),
            evidence=(EVIDENCE,),
            admissible_evidence_ids=(EVIDENCE.id,),
            remaining_budget=ResearchBudget(
                wall_clock_seconds=3600.0, model_tokens=100_000
            ),
        )

    def request(self, context: RoleContext, catalog) -> ModelRequest:  # type: ignore[no-untyped-def]
        return ModelRequest(
            model="scripted",
            instruction="plan",
            messages=(
                Message(
                    role=MessageRole.USER,
                    content=render_planning_context(context, catalog),
                ),
            ),
            schema=PLANNING_SCHEMA,
        )

    def test_the_sharpening_decision_passes_the_gate(
        self, tmp_path: Path
    ) -> None:
        catalog = catalog_for((PREDICTION,), trainer="stub")
        context = self.context()

        payload = json.loads(
            _planning_decision(self.request(context, catalog))
        )

        assert payload["action"] == "new_experiment"
        assert payload["prediction_metric"] == ENCODER_METRIC
        assert (
            check_decision(payload, context=context, catalog=catalog) == ()
        )

    def test_the_stop_decision_passes_the_gate(self, tmp_path: Path) -> None:
        catalog = catalog_for((PREDICTION,), trainer="stub")
        sharpened = Prediction(
            hypothesis_id=HYPOTHESIS.id,
            condition="at a fresh seed",
            metric=ENCODER_METRIC,
            comparator=Comparator.GREATER_OR_EQUAL,
            threshold=0.01,
            expectation="the floor holds",
        )
        context = RoleContext(
            questions=(QUESTION,),
            hypotheses=(HYPOTHESIS,),
            predictions=(PREDICTION, sharpened),
            experiments=(SPEC,),
            evidence=(EVIDENCE,),
            admissible_evidence_ids=(EVIDENCE.id,),
        )

        payload = json.loads(
            _planning_decision(self.request(context, catalog))
        )

        assert payload["action"] == "stop"
        assert payload["stop_reason"] == "question_resolved"
        assert payload["evidence_ids"] == [EVIDENCE.id]
        assert (
            check_decision(payload, context=context, catalog=catalog) == ()
        )


class TestReplicateFirstUnderRealCosts:
    """The tie-break trap the playbook exists to close: with a real
    trainer's cost estimate, synthesis outbids replication at equal
    MEDIUM value and the claim would be judged at n=1 — permanently.
    The playbook's replication-gap advice boosts REPLICATE to HIGH."""

    def costly_frontier(self) -> ResearchFrontier:
        expensive = ExperimentSpec(
            prediction_id=PREDICTION.id,
            objective="measure the contrast",
            procedure="run the template",
            metrics=(ENCODER_METRIC,),
            seeds=(11, 23, 47, 71, 83),
            estimated_cost=ResourceCost(wall_clock_seconds=600.0),
        )
        return frontier(
            replication_gaps=(expensive,),
            unsynthesized_evidence=("ev_1",),
            recent_results=(
                ResultRef(
                    result_id="res_1",
                    spec_id=expensive.id,
                    status=ExperimentStatus.COMPLETED,
                ),
            ),
        )

    def test_without_advice_synthesis_outbids_replication(self) -> None:
        from autonomous_research_lab.orchestration.director import (
            RuleBasedFrontierDirector,
        )

        chosen = RuleBasedFrontierDirector().deliberate(
            self.costly_frontier()
        )
        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.SYNTHESIZE_FINDING
        )

    def test_the_playbook_restores_replicate_first(self) -> None:
        from autonomous_research_lab.orchestration.director import (
            RuleBasedFrontierDirector,
        )
        from autonomous_research_lab.runtime.playbook import (
            EmpiricalMLPlaybook,
        )

        watched = self.costly_frontier()
        advice = EmpiricalMLPlaybook().advise(watched)
        assert advice is not None

        chosen = RuleBasedFrontierDirector().deliberate(
            watched, advice=advice
        )

        assert chosen.selected is not None
        assert (
            chosen.selected.action.action_type
            is ResearchActionType.REPLICATE
        )
