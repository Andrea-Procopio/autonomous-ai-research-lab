"""The run directive: the one bounded input of starting a research run.

A directive names the admission record whose state it funds
(:func:`~.door.require_admitted_state_for_run` is the only entrance) and
the authorization that pays for it. There is no "latest admission" and
no "current grant" anywhere: an admission this directive does not name
does not exist for this run.

A directive is content-addressed, and its identity is what makes a run
replayable rather than duplicated: re-running a completed directive
returns the run it already started, at no further cost and with no
second grant. Two runs over one admission are therefore two *stated*
runs — a second grant, or a second label saying how this run differs —
never an accident of running the same command twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..core.ids import content_id

MAX_LABEL_CHARS: Final = 120
"""A label names the run for a human reading a directory listing. It is
not a description, and nothing reads it as one."""


@dataclass(frozen=True, slots=True)
class RunDirective:
    """One bounded run assignment."""

    admission_record_id: str
    """The one door in: the funded state descends from this admission's
    state, and from no other."""

    authorization_id: str
    """The grant that funds it. The door refuses an authorization issued
    against a different admission."""

    label: str
    """What distinguishes this run from another over the same admission —
    an operator statement, required, so that a second run is deliberate.
    Two directives that agree on everything including the label are one
    directive, and the second is a replay."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        for name, value in (
            ("admission_record_id", self.admission_record_id),
            ("authorization_id", self.authorization_id),
        ):
            if not value.strip():
                raise ValueError(f"a run directive must name its {name}")
        if not self.label.strip():
            raise ValueError(
                "a run directive must carry a label saying what this run "
                "is; a second run over one admission is a deliberate act"
            )
        if len(self.label) > MAX_LABEL_CHARS:
            raise ValueError(
                f"label must be at most {MAX_LABEL_CHARS} characters, got "
                f"{len(self.label)}"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "rdir",
                    self.admission_record_id,
                    self.authorization_id,
                    self.label,
                ),
            )
