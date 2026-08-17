"""Deterministic role routing: obvious assignments cost zero model calls.

The runtime has exactly three seats:

* **scientist** (``RESEARCH_DIRECTOR``) — hypotheses, predictions,
  experiment design, synthesis: the thinking work the director itself owns;
* **executor** (``RESEARCH_ENGINEER``) — implementing and running
  experiments: short-lived, isolated, given only its assignment;
* **critic / analyst** (``RESULT_ANALYST``) — analysis, verification,
  epistemic assessment.

Which seat performs which action type is a static table. Nothing about the
mapping is uncertain today, so nothing about it should be inferred:
model-based routing is a Tier-1 cost with no Tier-1 question to answer. The
richer machinery — :class:`~autonomous_research_lab.roles.base.RoleSuitability`
and :class:`~autonomous_research_lab.orchestration.assignment.RegistryAssigner`
— stays in the architecture for the day routing has calibration data behind
it; nothing here depends on it.

:func:`expected_proposals` is the other half of the deterministic contract:
which proposal kinds each action type may legitimately return, checked
mechanically against every proposal a role hands back.
"""

from __future__ import annotations

from typing import Final

from ..core.actions import ResearchActionType
from ..core.proposals import ProposalKind
from ..roles.base import RoleName

_A: Final = ResearchActionType
_SCIENTIST: Final = RoleName.RESEARCH_DIRECTOR
_EXECUTOR: Final = RoleName.RESEARCH_ENGINEER
_CRITIC: Final = RoleName.RESULT_ANALYST

STATIC_ROUTES: Final[dict[ResearchActionType, RoleName]] = {
    _A.SEARCH_LITERATURE: _SCIENTIST,
    _A.GENERATE_HYPOTHESIS: _SCIENTIST,
    _A.REFINE_HYPOTHESIS: _SCIENTIST,
    _A.DERIVE_PREDICTION: _SCIENTIST,
    _A.DESIGN_EXPERIMENT: _SCIENTIST,
    _A.EXPLORE_ALTERNATIVE: _SCIENTIST,
    _A.SYNTHESIZE_FINDING: _SCIENTIST,
    _A.STOP_INVESTIGATION: _SCIENTIST,
    _A.IMPLEMENT: _EXECUTOR,
    _A.DEBUG: _EXECUTOR,
    _A.RUN_EXPERIMENT: _EXECUTOR,
    _A.REPLICATE: _EXECUTOR,
    _A.TEST_BASELINE: _EXECUTOR,
    _A.SCALE_EXPERIMENT: _EXECUTOR,
    _A.ANALYZE: _CRITIC,
    _A.FALSIFY: _CRITIC,
    _A.ASSESS_CLAIM: _CRITIC,
}


def route(action_type: ResearchActionType) -> RoleName:
    """The seat that performs ``action_type``. Total over the enum, enforced
    by a test — an unroutable action type is a programming error, not a
    runtime decision."""
    return STATIC_ROUTES[action_type]


_PROPOSALS: Final[dict[ResearchActionType, frozenset[ProposalKind]]] = {
    _A.SEARCH_LITERATURE: frozenset({ProposalKind.QUESTION, ProposalKind.EVIDENCE}),
    _A.GENERATE_HYPOTHESIS: frozenset({ProposalKind.HYPOTHESIS}),
    _A.REFINE_HYPOTHESIS: frozenset({ProposalKind.HYPOTHESIS}),
    _A.DERIVE_PREDICTION: frozenset({ProposalKind.PREDICTION}),
    _A.DESIGN_EXPERIMENT: frozenset({ProposalKind.EXPERIMENT}),
    _A.EXPLORE_ALTERNATIVE: frozenset(
        {ProposalKind.HYPOTHESIS, ProposalKind.PREDICTION}
    ),
    _A.SYNTHESIZE_FINDING: frozenset({ProposalKind.CLAIM}),
    _A.STOP_INVESTIGATION: frozenset(),
    _A.IMPLEMENT: frozenset({ProposalKind.EXPERIMENT}),
    _A.DEBUG: frozenset({ProposalKind.RESULT, ProposalKind.EVIDENCE}),
    _A.RUN_EXPERIMENT: frozenset({ProposalKind.RESULT}),
    _A.REPLICATE: frozenset({ProposalKind.RESULT}),
    _A.TEST_BASELINE: frozenset({ProposalKind.RESULT}),
    _A.SCALE_EXPERIMENT: frozenset({ProposalKind.RESULT}),
    _A.ANALYZE: frozenset(
        {ProposalKind.EVIDENCE, ProposalKind.CLAIM, ProposalKind.ASSESSMENT}
    ),
    _A.FALSIFY: frozenset({ProposalKind.EVIDENCE, ProposalKind.ASSESSMENT}),
    _A.ASSESS_CLAIM: frozenset({ProposalKind.ASSESSMENT}),
}


def expected_proposals(action_type: ResearchActionType) -> frozenset[ProposalKind]:
    """The output contract for an invocation performing ``action_type``."""
    return _PROPOSALS[action_type]
