"""Research questions -- the unit of scientific intent above a hypothesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ids import content_id


class QuestionStatus(StrEnum):
    OPEN = "open"
    ANSWERED = "answered"
    DEFERRED = "deferred"
    ABANDONED = "abandoned"


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    text: str
    importance: str = ""
    """Why answering this would matter scientifically. Recorded because a
    system that cannot say why a question matters cannot prioritise between
    questions."""

    status: QuestionStatus = QuestionStatus.OPEN
    parent_id: str | None = None
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", content_id("q", self.text, self.parent_id))

    def with_status(self, status: QuestionStatus) -> ResearchQuestion:
        return ResearchQuestion(
            text=self.text,
            importance=self.importance,
            status=status,
            parent_id=self.parent_id,
            id=self.id,
        )
