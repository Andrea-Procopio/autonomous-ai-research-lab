"""Read models over accumulated scientific knowledge.

Today: the claim-evidence graph, and the shape (only the shape) of a
cross-project lesson. Persistent institutional memory is deliberately not
built yet; it should be designed against real research trajectories rather
than guessed at.
"""

from .graph import ClaimEvidence, ClaimEvidenceGraph
from .lessons import LabLesson

__all__ = ["ClaimEvidence", "ClaimEvidenceGraph", "LabLesson"]
