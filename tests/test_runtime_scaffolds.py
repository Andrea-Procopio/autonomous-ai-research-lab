"""The deliberately-minimal scaffolds: playbook advice, held-out hooks,
lessons, and the config's guardrails."""

from __future__ import annotations

import dataclasses

import pytest

from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.core.state import ResearchState
from autonomous_research_lab.knowledge.lessons import LabLesson
from autonomous_research_lab.runtime.config import RuntimeConfig
from autonomous_research_lab.runtime.evaluators import (
    EvaluationHooks,
    HeldOutAccessError,
)
from autonomous_research_lab.runtime.frontier import (
    Contradiction,
    ResearchFrontier,
    build_frontier,
)
from autonomous_research_lab.runtime.playbook import EmpiricalMLPlaybook

_QUESTION = ResearchQuestion(text="Is the stream fair?")
_HYPOTHESIS = Hypothesis(
    statement="The stream is biased.", question_id=_QUESTION.id
)


def _frontier(**overrides: object) -> ResearchFrontier:
    state = (
        ResearchState(objective="fairness")
        .upsert_question(_QUESTION)
        .upsert_hypothesis(_HYPOTHESIS)
    )
    frontier = build_frontier(state)
    if overrides:
        frontier = dataclasses.replace(frontier, **overrides)  # type: ignore[arg-type]
    return frontier


class _FixedEvaluator:
    def __init__(self, name: str, value: float) -> None:
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    def score(self, result: ExperimentResult) -> float:
        return self._value


_RESULT = ExperimentResult(
    spec_id="exp_scaffold",
    job_id="job_eval",
    status=ExperimentStatus.COMPLETED,
    command=("run",),
    environment=Environment(python_version="3.11", platform="test"),
    metrics={"heads_rate": 0.5},
    exit_code=0,
)


def test_playbook_advises_and_never_mandates() -> None:
    playbook = EmpiricalMLPlaybook()

    nothing_run = _frontier()
    advice = playbook.advise(nothing_run)
    assert advice is not None
    assert advice.stage.name == "establish_viability"

    # A contradiction outranks everything else.
    conflicted = _frontier(
        contradictions=(
            Contradiction(
                subject_kind="prediction", subject_id="pred_x", detail="mixed"
            ),
        )
    )
    conflicted_advice = playbook.advise(conflicted)
    assert conflicted_advice is not None
    assert conflicted_advice.stage.name == "investigate_mechanism"

    # Advice is data, not control flow: nothing here can force an action.
    assert not hasattr(advice, "action")
    assert not hasattr(playbook, "enforce")


def test_development_scoring_is_free_but_held_out_needs_a_release() -> None:
    hooks = EvaluationHooks(
        development=_FixedEvaluator("dev", 0.5),
        held_out=_FixedEvaluator("held", 0.7),
    )
    assert hooks.score_development(_RESULT) == 0.5
    before = hooks.held_out_accesses
    assert before == ()

    with pytest.raises(HeldOutAccessError):
        hooks.score_held_out(_RESULT, released_by="")

    score = hooks.score_held_out(_RESULT, released_by="andrea, final eval")
    assert score == 0.7
    accesses = hooks.held_out_accesses
    assert len(accesses) == 1
    assert accesses[0].released_by == "andrea, final eval"
    assert accesses[0].result_id == _RESULT.id


def test_lessons_demand_scope_and_carry_their_evidence() -> None:
    lesson = LabLesson(
        statement="Seeded replications with < 3 seeds are underpowered here.",
        scope="coin-bias family experiments",
        evidence_ids=("ev_1", "ev_2"),
        confidence=0.6,
        source_projects=("demo",),
    )
    assert lesson.id.startswith("les_")
    assert lesson == LabLesson(
        statement=lesson.statement,
        scope=lesson.scope,
        evidence_ids=lesson.evidence_ids,
        confidence=0.6,
        source_projects=("demo",),
    )

    with pytest.raises(ValueError, match="scope"):
        LabLesson(statement="Everything is fine.", scope="  ")


def test_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        RuntimeConfig(synthesis_every=0)
    with pytest.raises(ValueError):
        RuntimeConfig(max_candidates=0)
