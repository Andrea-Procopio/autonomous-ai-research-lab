"""Institutional memory, scaffold only: the shape of a reusable lesson.

Two very different things will eventually deserve the name "memory":

* **raw trajectory history** — every decision, attempt and result, already
  persisted by the trajectory log and state snapshots; cheap, complete, and
  unvetted;
* **verified reusable lessons** — the small residue of that history that has
  been checked and is worth carrying into other projects.

This module defines only the second thing's shape. Nothing populates lessons
automatically — promotion from trajectory to lesson is a curation act that
needs real trajectories to curate, and an automatic promoter would be a
machine for laundering one noisy run into cross-project dogma. No storage, no
retrieval, no vectors; those are decisions to make against real usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.ids import content_id


@dataclass(frozen=True, slots=True)
class LabLesson:
    """One verified, scoped, evidence-backed lesson.

    Content identity: the same lesson learned twice is the same lesson.
    """

    statement: str
    scope: str
    """Where the lesson applies. An unscoped lesson is a superstition."""

    evidence_ids: tuple[str, ...] = ()
    """The recorded evidence that backs it — a lesson with no evidence
    references is an opinion and should stay in someone's head."""

    confidence: float | None = None
    known_exceptions: tuple[str, ...] = ()
    source_projects: tuple[str, ...] = ()
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("lesson statement must be non-empty")
        if not self.scope.strip():
            raise ValueError("lesson requires a scope")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "les",
                    self.statement,
                    self.scope,
                    self.evidence_ids,
                    self.known_exceptions,
                ),
            )
