"""Read models over accumulated scientific knowledge.

Today: the claim-evidence graph. Later: persistent cross-project memory —
which methods worked, which failure modes recur, which questions were already
answered and by what. That memory is deliberately not started yet; it should be
designed against real research trajectories rather than guessed at.
"""

from .graph import ClaimEvidence, ClaimEvidenceGraph

__all__ = ["ClaimEvidence", "ClaimEvidenceGraph"]
