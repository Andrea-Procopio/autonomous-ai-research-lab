"""Evidence: factual observations grounded in a recorded result.

Evidence sits between raw results and interpreted claims.

* An :class:`~.experiment.ExperimentResult` is what the machine emitted.
* An :class:`Evidence` is a factual reading of that result, always pointing
  back at it ("heads_rate = 0.503 over 4000 draws, seed 7").
* A :class:`~.claim.Claim` is an interpretation, and never stores its own
  numbers -- it links to evidence.

The separation exists so that reinterpreting a result can never destroy it, and
so that any claim can be walked back to the process that produced its support.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .ids import content_id
from .types import freeze_mapping


class EvidenceKind(StrEnum):
    MEASUREMENT = "measurement"
    NULL_RESULT = "null_result"
    FAILURE = "failure"
    """The implementation or run failed. Recorded because a failure constrains
    what can be concluded, and because repeated failures are themselves a
    finding."""

    REPLICATION = "replication"
    BASELINE = "baseline"
    ANOMALY = "anomaly"


@dataclass(frozen=True, slots=True)
class Evidence:
    result_id: str
    spec_id: str
    kind: EvidenceKind
    observation: str
    """A factual restatement of what was measured, not what it means. Wording
    that draws a conclusion belongs in a claim."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    """The subset of the result's metrics this evidence reads, copied so the
    reading is pinned even if later evidence reads other metrics."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id("ev", self.result_id, self.kind, self.observation),
            )
