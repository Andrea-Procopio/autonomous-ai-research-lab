"""Content-addressed persistence of ``ResearchState`` snapshots.

Layout, under a run directory::

    <root>/
    └── states/
        └── <state_id>.json      one deterministic snapshot per state

State ids are content-derived, so the layout deduplicates by construction:
persisting the same state twice writes one file, and two branches that reach
identical content share a snapshot. Trajectory records name state ids
(``state_before_id`` / ``state_after_id``), so a decision log plus this store
reconstructs every decision's before/after states offline.

Serialization is deterministic — field order from the dataclass definitions,
keys sorted, no timestamps — and loading verifies integrity by recomputing
the content id from the reconstructed state: a snapshot that no longer hashes
to its filename is corrupt and says so, rather than quietly resurrecting a
different state.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.actions import ResearchAction, ResearchActionType
from ..core.assessment import AssessmentVerdict, EpistemicAssessment
from ..core.attempt import ActionAttempt, ActionOutcome, AttemptStatus
from ..core.budget import ResearchBudget, ResourceCost
from ..core.claim import Claim, EvidenceLink, EvidenceRelation
from ..core.experiment import ExperimentSpec, ExperimentStatus, ResultRef
from ..core.hypothesis import Hypothesis
from ..core.prediction import Comparator, Consistency, Prediction, PredictionTest
from ..core.question import QuestionStatus, ResearchQuestion
from ..core.serialize import to_jsonable
from ..core.state import ResearchState


class SnapshotError(RuntimeError):
    """Raised when a snapshot is missing, malformed, or fails verification."""


class FileStateStore:
    """Local filesystem store: ``persist(state) -> reload(state.id)``."""

    def __init__(self, root: Path | str) -> None:
        self._states = Path(root) / "states"
        self._states.mkdir(parents=True, exist_ok=True)

    def path_for(self, state_id: str) -> Path:
        return self._states / f"{state_id}.json"

    def persist(self, state: ResearchState) -> Path:
        """Write ``state`` and return its snapshot path. Identical content
        deduplicates; a snapshot whose bytes differ from what this state
        serializes to (a truncated write, a tampered file) is refused loudly
        rather than silently kept. The write goes through a temporary file
        and an atomic rename, so a crash mid-write can never leave a partial
        snapshot under the state's name."""
        path = self.path_for(state.id)
        payload = serialize_state(state)
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise SnapshotError(
                    f"snapshot {path.name} exists with different content; "
                    f"snapshots are never rewritten"
                )
            return path
        scratch = path.with_suffix(".json.tmp")
        scratch.write_text(payload, encoding="utf-8")
        try:
            scratch.replace(path)
        finally:
            scratch.unlink(missing_ok=True)
        return path

    def load(self, state_id: str) -> ResearchState:
        path = self.path_for(state_id)
        if not path.exists():
            raise SnapshotError(f"no snapshot for state {state_id}")
        state = deserialize_state(path.read_text(encoding="utf-8"))
        if state.id != state_id:
            raise SnapshotError(
                f"snapshot {path.name} reconstructs to state {state.id}; "
                f"the file is corrupt or the schema drifted"
            )
        return state

    def state_ids(self) -> tuple[str, ...]:
        return tuple(sorted(p.stem for p in self._states.glob("*.json")))


def serialize_state(state: ResearchState) -> str:
    """Deterministic JSON: same state content, same bytes."""
    return json.dumps(to_jsonable(state), sort_keys=True, indent=2) + "\n"


def deserialize_state(text: str) -> ResearchState:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"snapshot is not valid JSON: {exc}") from exc
    raw = _dict(payload, "state")
    stored_id = _str(_get(raw, "id", "state"), "state.id")
    state = _state(raw)
    if state.id != stored_id:
        raise SnapshotError(
            f"snapshot claims id {stored_id} but reconstructs to {state.id}"
        )
    return state


# -- reconstruction ----------------------------------------------------------
# Boundary code, deliberately explicit: every field is checked as it is read,
# and the state id is recomputed from content (the id field is stripped before
# construction) so codec drift surfaces as an error instead of a wrong state.


def _state(raw: dict[str, object]) -> ResearchState:
    return ResearchState(
        objective=_str(_get(raw, "objective", "state"), "state.objective"),
        questions=tuple(
            _question(_dict(q, "question")) for q in _list(raw.get("questions"))
        ),
        hypotheses=tuple(
            _hypothesis(_dict(h, "hypothesis"))
            for h in _list(raw.get("hypotheses"))
        ),
        predictions=tuple(
            _prediction(_dict(p, "prediction"))
            for p in _list(raw.get("predictions"))
        ),
        experiments=tuple(
            _spec(_dict(e, "experiment")) for e in _list(raw.get("experiments"))
        ),
        results=tuple(
            _result_ref(_dict(r, "result ref")) for r in _list(raw.get("results"))
        ),
        evidence_ids=_str_tuple(raw.get("evidence_ids"), "state.evidence_ids"),
        prediction_tests=tuple(
            _prediction_test(_dict(t, "prediction test"))
            for t in _list(raw.get("prediction_tests"))
        ),
        claims=tuple(_claim(_dict(c, "claim")) for c in _list(raw.get("claims"))),
        evidence_links=tuple(
            _link(_dict(link, "evidence link"))
            for link in _list(raw.get("evidence_links"))
        ),
        assessments=tuple(
            _assessment(_dict(a, "assessment"))
            for a in _list(raw.get("assessments"))
        ),
        attempts=tuple(
            _attempt(_dict(a, "attempt")) for a in _list(raw.get("attempts"))
        ),
        budget=_budget(_dict(_get(raw, "budget", "state"), "budget")),
        history=tuple(
            _action(_dict(a, "action")) for a in _list(raw.get("history"))
        ),
        parent_id=_opt_str(raw.get("parent_id"), "state.parent_id"),
    )


def _question(raw: dict[str, object]) -> ResearchQuestion:
    return ResearchQuestion(
        text=_str(_get(raw, "text", "question"), "question.text"),
        importance=_str(raw.get("importance", ""), "question.importance"),
        status=QuestionStatus(_str(_get(raw, "status", "question"), "status")),
        parent_id=_opt_str(raw.get("parent_id"), "question.parent_id"),
        id=_str(_get(raw, "id", "question"), "question.id"),
    )


def _hypothesis(raw: dict[str, object]) -> Hypothesis:
    return Hypothesis(
        statement=_str(_get(raw, "statement", "hypothesis"), "statement"),
        rationale=_str(raw.get("rationale", ""), "hypothesis.rationale"),
        assumptions=_str_tuple(raw.get("assumptions"), "hypothesis.assumptions"),
        question_id=_opt_str(raw.get("question_id"), "hypothesis.question_id"),
        parent_id=_opt_str(raw.get("parent_id"), "hypothesis.parent_id"),
        id=_str(_get(raw, "id", "hypothesis"), "hypothesis.id"),
    )


def _prediction(raw: dict[str, object]) -> Prediction:
    return Prediction(
        hypothesis_id=_str(_get(raw, "hypothesis_id", "prediction"), "hypothesis_id"),
        condition=_str(_get(raw, "condition", "prediction"), "condition"),
        metric=_str(_get(raw, "metric", "prediction"), "metric"),
        comparator=Comparator(
            _str(_get(raw, "comparator", "prediction"), "comparator")
        ),
        threshold=_number(_get(raw, "threshold", "prediction"), "threshold"),
        tolerance=_number(raw.get("tolerance", 0.0), "tolerance"),
        expectation=_str(raw.get("expectation", ""), "expectation"),
        scope=_str(raw.get("scope", ""), "prediction.scope"),
        id=_str(_get(raw, "id", "prediction"), "prediction.id"),
    )


def _prediction_test(raw: dict[str, object]) -> PredictionTest:
    return PredictionTest(
        prediction_id=_str(_get(raw, "prediction_id", "test"), "prediction_id"),
        result_id=_str(_get(raw, "result_id", "test"), "result_id"),
        metric=_str(_get(raw, "metric", "test"), "test.metric"),
        observed=_opt_number(raw.get("observed"), "test.observed"),
        consistency=Consistency(_str(_get(raw, "consistency", "test"), "consistency")),
        detail=_str(raw.get("detail", ""), "test.detail"),
        id=_str(_get(raw, "id", "test"), "test.id"),
    )


def _spec(raw: dict[str, object]) -> ExperimentSpec:
    return ExperimentSpec(
        prediction_id=_str(_get(raw, "prediction_id", "spec"), "prediction_id"),
        objective=_str(_get(raw, "objective", "spec"), "spec.objective"),
        procedure=_str(_get(raw, "procedure", "spec"), "spec.procedure"),
        metrics=_str_tuple(_get(raw, "metrics", "spec"), "spec.metrics"),
        baselines=_str_tuple(raw.get("baselines"), "spec.baselines"),
        controls=_str_tuple(raw.get("controls"), "spec.controls"),
        seeds=tuple(_int(s, "spec.seeds") for s in _list(raw.get("seeds"))),
        estimated_cost=_cost(
            _dict(_get(raw, "estimated_cost", "spec"), "estimated_cost")
        ),
        id=_str(_get(raw, "id", "spec"), "spec.id"),
    )


def _result_ref(raw: dict[str, object]) -> ResultRef:
    return ResultRef(
        result_id=_str(_get(raw, "result_id", "result ref"), "result_id"),
        spec_id=_str(_get(raw, "spec_id", "result ref"), "spec_id"),
        status=ExperimentStatus(_str(_get(raw, "status", "result ref"), "status")),
    )


def _claim(raw: dict[str, object]) -> Claim:
    return Claim(
        statement=_str(_get(raw, "statement", "claim"), "claim.statement"),
        scope=_str(raw.get("scope", ""), "claim.scope"),
        hypothesis_id=_opt_str(raw.get("hypothesis_id"), "claim.hypothesis_id"),
        id=_str(_get(raw, "id", "claim"), "claim.id"),
    )


def _link(raw: dict[str, object]) -> EvidenceLink:
    return EvidenceLink(
        claim_id=_str(_get(raw, "claim_id", "link"), "link.claim_id"),
        evidence_id=_str(_get(raw, "evidence_id", "link"), "link.evidence_id"),
        relation=EvidenceRelation(_str(_get(raw, "relation", "link"), "relation")),
        rationale=_str(raw.get("rationale", ""), "link.rationale"),
        id=_str(_get(raw, "id", "link"), "link.id"),
    )


def _assessment(raw: dict[str, object]) -> EpistemicAssessment:
    return EpistemicAssessment(
        subject_id=_str(_get(raw, "subject_id", "assessment"), "subject_id"),
        verdict=AssessmentVerdict(_str(_get(raw, "verdict", "assessment"), "verdict")),
        method=_str(_get(raw, "method", "assessment"), "assessment.method"),
        evidence_ids=_str_tuple(raw.get("evidence_ids"), "assessment.evidence_ids"),
        confidence=_opt_number(raw.get("confidence"), "assessment.confidence"),
        scope=_str(raw.get("scope", ""), "assessment.scope"),
        rationale=_str(raw.get("rationale", ""), "assessment.rationale"),
        supersedes=_opt_str(raw.get("supersedes"), "assessment.supersedes"),
        id=_str(_get(raw, "id", "assessment"), "assessment.id"),
    )


def _action(raw: dict[str, object]) -> ResearchAction:
    return ResearchAction(
        action_type=ResearchActionType(
            _str(_get(raw, "action_type", "action"), "action_type")
        ),
        rationale=_str(_get(raw, "rationale", "action"), "action.rationale"),
        targets=_str_tuple(raw.get("targets"), "action.targets"),
        id=_str(_get(raw, "id", "action"), "action.id"),
    )


def _outcome(raw: dict[str, object]) -> ActionOutcome:
    return ActionOutcome(
        status=AttemptStatus(_str(_get(raw, "status", "outcome"), "outcome.status")),
        produced=_str_tuple(raw.get("produced"), "outcome.produced"),
        error=_opt_str(raw.get("error"), "outcome.error"),
        actual_cost=_cost(_dict(_get(raw, "actual_cost", "outcome"), "actual_cost")),
    )


def _attempt(raw: dict[str, object]) -> ActionAttempt:
    outcome_raw = raw.get("outcome")
    return ActionAttempt(
        action=_action(_dict(_get(raw, "action", "attempt"), "attempt.action")),
        status=AttemptStatus(_str(_get(raw, "status", "attempt"), "attempt.status")),
        outcome=None
        if outcome_raw is None
        else _outcome(_dict(outcome_raw, "attempt.outcome")),
        id=_str(_get(raw, "id", "attempt"), "attempt.id"),
    )


def _cost(raw: dict[str, object]) -> ResourceCost:
    return ResourceCost(
        wall_clock_seconds=_number(
            raw.get("wall_clock_seconds", 0.0), "cost.wall_clock_seconds"
        ),
        gpu_hours=_number(raw.get("gpu_hours", 0.0), "cost.gpu_hours"),
        usd=_number(raw.get("usd", 0.0), "cost.usd"),
        model_tokens=_int(raw.get("model_tokens", 0), "cost.model_tokens"),
    )


def _budget(raw: dict[str, object]) -> ResearchBudget:
    return ResearchBudget(
        wall_clock_seconds=_number(
            raw.get("wall_clock_seconds", 0.0), "budget.wall_clock_seconds"
        ),
        gpu_hours=_number(raw.get("gpu_hours", 0.0), "budget.gpu_hours"),
        usd=_number(raw.get("usd", 0.0), "budget.usd"),
        model_tokens=_int(raw.get("model_tokens", 0), "budget.model_tokens"),
    )


# -- narrowing helpers -------------------------------------------------------


def _get(raw: dict[str, object], key: str, where: str) -> object:
    if key not in raw:
        raise SnapshotError(f"{where} is missing required field {key!r}")
    return raw[key]


def _dict(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SnapshotError(f"{where} must be an object, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SnapshotError(f"expected a list, got {type(value).__name__}")
    return value


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise SnapshotError(f"{where} must be a string, got {type(value).__name__}")
    return value


def _opt_str(value: object, where: str) -> str | None:
    return None if value is None else _str(value, where)


def _number(value: object, where: str) -> float:
    """Accept int or float, preserving the exact value that was written —
    content ids distinguish ``0`` from ``0.0``, so no coercion happens here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotError(f"{where} must be a number, got {type(value).__name__}")
    return value


def _opt_number(value: object, where: str) -> float | None:
    return None if value is None else _number(value, where)


def _int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotError(f"{where} must be an integer, got {type(value).__name__}")
    return value


def _str_tuple(value: object, where: str) -> tuple[str, ...]:
    return tuple(_str(item, where) for item in _list(value))
