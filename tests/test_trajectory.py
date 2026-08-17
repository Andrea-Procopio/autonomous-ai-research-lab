"""Trajectory logging: the decision tuple survives, serialized, from step one.

What later scientific evaluation needs per decision:
state before, every candidate with its utility estimate and the method that
produced it, what was selected and by which policy, the attempt, its outcome,
actual cost, and the state after. These tests pin that the record carries all
of it and that the JSONL round-trips.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from autonomous_research_lab.core.attempt import ActionOutcome, AttemptStatus
from autonomous_research_lab.core.budget import ResearchBudget, ResourceCost
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.orchestration.candidates import RuleBasedCandidateGenerator
from autonomous_research_lab.orchestration.director import ResearchDirector
from autonomous_research_lab.orchestration.evaluation import HeuristicUtilityEvaluator
from autonomous_research_lab.orchestration.trajectory import JsonlTrajectoryLogger
from autonomous_research_lab.search.policy import GreedySearchPolicy
from examples.minimal_loop import run_minimal_loop


def director() -> ResearchDirector:
    return ResearchDirector(
        generator=RuleBasedCandidateGenerator(),
        evaluator=HeuristicUtilityEvaluator(),
        policy=GreedySearchPolicy(),
    )


def funded_state() -> ResearchState:
    return ResearchState(
        objective="o",
        budget=ResearchBudget(wall_clock_seconds=1e6, usd=1e3, model_tokens=10**7),
    )


class TestDecisionRecord:
    def test_records_every_candidate_with_its_utility(self) -> None:
        decision = director().decide(funded_state())
        record = decision.record

        assert record.state_before_id == funded_state().id
        assert record.evaluated, "no candidates recorded"
        for evaluated in record.evaluated:
            assert evaluated.utility.method == "heuristic:v0"
        assert record.generator == "rule-based:v1"
        assert record.evaluator == "heuristic:v0"
        assert record.policy == "greedy:v0"
        assert record.selected_action_id is not None
        assert record.assigned_role is None  # no roles exist yet; slot is live

    def test_completion_preserves_identity(self) -> None:
        decision = director().decide(funded_state())
        outcome = ActionOutcome(
            status=AttemptStatus.SUCCEEDED, actual_cost=ResourceCost(usd=1.0)
        )
        completed = decision.record.completed(
            attempt_id="att_x",
            outcome=outcome,
            state_after_id="st_after",
            assigned_role="skeptic",
        )
        assert completed.id == decision.record.id
        assert completed.outcome == outcome
        assert completed.state_after_id == "st_after"
        assert completed.assigned_role == "skeptic"

    def test_predicted_cost_reads_the_selected_candidates_estimate(self) -> None:
        decision = director().decide(funded_state())
        assert decision.selected is not None
        assert (
            decision.record.predicted_cost
            == decision.selected.utility.expected_cost
        )


class TestJsonlLogger:
    def test_log_and_read_round_trip(self, tmp_path: Path) -> None:
        logger = JsonlTrajectoryLogger(tmp_path / "trajectory.jsonl")
        decision = director().decide(funded_state())
        logger.log(decision.record)

        (row,) = logger.read()
        assert row["id"] == decision.record.id
        assert row["state_before_id"] == funded_state().id
        assert row["policy"] == "greedy:v0"
        assert "logged_at" in row
        assert "assigned_role" in row  # serialized even while always None
        evaluated = row["evaluated"]
        assert isinstance(evaluated, list) and evaluated
        first = evaluated[0]
        assert isinstance(first, dict)
        utility = first["utility"]
        assert isinstance(utility, dict)
        assert utility["method"] == "heuristic:v0"

    def test_full_loop_logs_every_decision(self, tmp_path: Path) -> None:
        outcome = run_minimal_loop(tmp_path, log_path=tmp_path / "t.jsonl")

        lines = (tmp_path / "t.jsonl").read_text().strip().splitlines()
        rows = [json.loads(line) for line in lines]
        assert len(rows) == len(outcome.decisions)

        # Every non-terminal decision carries its outcome and both state ids —
        # the (s, {a_i, U_i}, a, o, s') tuple is fully present.
        for row in rows[:-1]:
            assert row["selected_action_id"] is not None
            assert row["outcome"] is not None
            assert row["outcome"]["status"] == "succeeded"
            assert row["state_before_id"] != row["state_after_id"]

        # Consecutive decisions chain: state_after of one is state_before of
        # the next.
        for previous, current in pairwise(rows):
            assert previous["state_after_id"] == current["state_before_id"]
