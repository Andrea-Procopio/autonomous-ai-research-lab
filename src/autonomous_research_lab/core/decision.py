"""The decision vocabulary: candidates, utilities, and the decision record.

Three questions, three separate functions, three separate objects:

    What could we do?          -> candidate generation -> ActionCandidate
    How valuable might it be?  -> utility evaluation   -> ActionUtility
    What do we pick, given
    uncertainty and resources? -> search policy        -> the selected action

The invariant: **scientific utility is not search policy.** A utility describes
why an action might be worth taking; a policy decides how to explore given
those descriptions. A bandit and a greedy policy consuming identical utilities
are different explorers, not different opinions about science.

``ActionUtility`` is multi-dimensional on purpose. Collapsing research value to
one scalar is itself a modelling decision — the wrong one to hard-code before
any calibration data exists. Policies that need a scalar perform their own
scalarization, explicitly, where it can be seen and criticised.

``DecisionRecord`` preserves the full decision tuple
``(state, {candidate_i, utility_i}, selected, outcome, state')`` — the data a
later scientific evaluation of this architecture needs. It is recorded from the
first run, not retrofitted, because trajectories that only exist for successful
designs cannot support claims about the design.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .actions import ResearchAction
from .attempt import ActionOutcome
from .budget import NO_COST, ResourceCost
from .ids import occurrence_id
from .serialize import Jsonable, to_jsonable


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """An available action, with the provenance of its availability."""

    action: ResearchAction
    generated_by: str
    """Name of the generator that surfaced this candidate."""


@dataclass(frozen=True, slots=True)
class ActionUtility:
    """A multi-dimensional estimate of an action's scientific value.

    Every dimension is optional; ``None`` means *not estimated*, which no
    consumer may silently read as zero. The estimate names its ``method`` so
    that utility quality can itself be evaluated later — heuristic, model
    judgment, and learned estimators all fit this shape.
    """

    expected_information_gain: float | None = None
    discrimination_value: float | None = None
    """How strongly the outcome would discriminate between live hypotheses —
    falsification potential."""

    importance: float | None = None
    novelty: float | None = None
    replication_value: float | None = None
    expected_success_probability: float | None = None
    expected_cost: ResourceCost = NO_COST
    estimate_uncertainty: float | None = None
    """The evaluator's own uncertainty about this estimate, in [0, 1]."""

    method: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        for name in ("expected_success_probability", "estimate_uncertainty"):
            bounded: float | None = getattr(self, name)
            if bounded is not None and not 0.0 <= bounded <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {bounded}")


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    candidate: ActionCandidate
    utility: ActionUtility

    @property
    def action(self) -> ResearchAction:
        return self.candidate.action


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One orchestration decision, preserved end to end.

    Occurrence identity: deciding twice from the same state is two decisions.
    Created when the decision is made, completed (via :meth:`completed`) once
    the outcome is known — completion preserves identity.
    """

    state_before_id: str
    evaluated: tuple[EvaluatedCandidate, ...]
    selected_action_id: str | None
    generator: str
    evaluator: str
    policy: str
    assigned_role: str | None = None
    """The role assigned to perform the selected action, when routing
    happened. Preserved so trajectories can later answer whether role
    specialization helps — a question about (action, role, outcome) triples."""

    attempt_id: str | None = None
    outcome: ActionOutcome | None = None
    state_after_id: str | None = None
    """With state snapshots persisted, ``state_before_id`` and
    ``state_after_id`` double as references into the state store, so the full
    (state, candidates, selection, outcome, state') tuple reconstructs
    offline."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", occurrence_id("dec"))

    def completed(
        self,
        *,
        attempt_id: str | None,
        outcome: ActionOutcome | None,
        state_after_id: str,
        assigned_role: str | None = None,
    ) -> DecisionRecord:
        return replace(
            self,
            attempt_id=attempt_id,
            outcome=outcome,
            state_after_id=state_after_id,
            assigned_role=assigned_role
            if assigned_role is not None
            else self.assigned_role,
        )

    @property
    def predicted_cost(self) -> ResourceCost:
        selected = next(
            (e for e in self.evaluated if e.action.id == self.selected_action_id),
            None,
        )
        return selected.utility.expected_cost if selected else NO_COST

    def to_jsonable(self) -> Jsonable:
        return to_jsonable(self)
