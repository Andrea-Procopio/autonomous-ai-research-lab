"""Orchestration: deciding, committing, and remembering why.

* ``candidates`` — what could be done (:class:`CandidateGenerator`);
* ``evaluation`` — how valuable each option might be (:class:`UtilityEvaluator`);
* ``director`` — one decision, wired from the two above plus a search policy;
* ``transitions`` — the only path from a role's proposal to a state change;
* ``assignment`` — which role performs a chosen action;
* ``trajectory`` — every decision preserved for later evaluation.
"""

from .assignment import RegistryAssigner, RoleAssigner
from .candidates import CandidateGenerator, RuleBasedCandidateGenerator
from .director import Decision, ResearchDirector
from .evaluation import HeuristicUtilityEvaluator, UtilityEvaluator
from .trajectory import JsonlTrajectoryLogger
from .transitions import TransitionError, commit

__all__ = [
    "CandidateGenerator",
    "Decision",
    "HeuristicUtilityEvaluator",
    "JsonlTrajectoryLogger",
    "RegistryAssigner",
    "ResearchDirector",
    "RoleAssigner",
    "RuleBasedCandidateGenerator",
    "TransitionError",
    "UtilityEvaluator",
    "commit",
]
