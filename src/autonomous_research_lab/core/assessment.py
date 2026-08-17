"""Epistemic assessment: judgment about a claim or hypothesis, as an object.

The architecture separates three things that fluent systems love to merge:

* **evidence** — factual readings of recorded results;
* **relations** — how each piece of evidence bears on a claim;
* **assessment** — whether, given the evidence considered, the claim should
  currently be believed, doubted, or suspended.

The first two are annotations of fact. The third is a *judgment*, and this
module makes judgments first-class: an :class:`EpistemicAssessment` names the
subject, the evidence it considered, the verdict, the confidence, the scope,
and — critically — the **method** that produced it. "One contradicting result
means refuted" is not a rule of this system; it would just be one (bad) method
that an assessment could cite and a reviewer could challenge.

Assessments are versioned by supersession, never edited: a change of mind is a
new assessment pointing at the one it replaces. The current standing of a
subject is the latest assessment targeting it, via
``ResearchState.current_assessment``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ids import content_id


class AssessmentVerdict(StrEnum):
    UNDETERMINED = "undetermined"
    """The evidence considered does not yet license a lean either way."""

    PLAUSIBLE = "plausible"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    """Non-trivial evidence points both ways and neither reading dominates."""

    REFUTED = "refuted"


@dataclass(frozen=True, slots=True)
class EpistemicAssessment:
    subject_id: str
    """Id of the claim or hypothesis being assessed (``clm_…`` or ``hyp_…``)."""

    verdict: AssessmentVerdict
    method: str
    """Who or what produced this judgment and how — ``"demo:prediction-check-v0"``,
    a role name, later a statistical procedure. Required: an assessment that
    cannot say how it was reached cannot be challenged, and a judgment that
    cannot be challenged is dogma."""

    evidence_ids: tuple[str, ...] = ()
    """The evidence actually considered — not necessarily all that exists.
    An assessment is only as good as what it looked at, and recording the
    difference is what lets a later pass find judgments made on stale data."""

    confidence: float | None = None
    """In [0, 1] when given. ``None`` means the method does not quantify its
    confidence, which is honest more often than a made-up number."""

    scope: str = ""
    rationale: str = ""
    supersedes: str | None = None
    """Id of the assessment this one replaces, forming a version chain."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("assessment requires a method")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "asm",
                    self.subject_id,
                    self.verdict,
                    self.method,
                    self.evidence_ids,
                    self.confidence,
                    self.scope,
                    self.rationale,
                    self.supersedes,
                ),
            )
