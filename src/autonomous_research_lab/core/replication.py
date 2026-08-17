"""Replication groups: which executions were testing the same thing.

Independent executions of one experiment are independent evidence — the store
and the identity rules already guarantee that two runs never collapse into one
record. What was missing is the converse question: *which results belong to
the same replication family?*

A :class:`ReplicationGroup` is defined by **protocol identity alone**: the
experiment spec and the configuration the process ran under. Two results with
the same spec and config are replications of each other — including when their
metrics disagree. Observed outcomes are deliberately excluded from the
grouping key: grouping by outcome would sort contradictory replications into
different families, which is precisely the moment they most need to be seen
together.

Deliberate exclusions, and why:

* **seed** — runs of one protocol under different seeds are statistical
  replications of the same thing; the seed stays visible on each member.
* **environment** — platform and commit are provenance, recorded per result;
  splitting families by machine would hide cross-machine disagreement, which
  is a finding, not a grouping error.
* **metrics** — outcomes never define what was tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .experiment import ExperimentResult
from .ids import content_id
from .types import ConfigValue


@dataclass(frozen=True, slots=True)
class ReplicationGroup:
    """The shared scientific/protocol identity of a set of executions.

    Content identity: the same spec under the same configuration is the same
    group, on every branch and in every run. Hashable, so results can be
    bucketed by group directly.
    """

    spec_id: str
    config_items: tuple[tuple[str, ConfigValue], ...] = ()
    """The run configuration as sorted ``(key, value)`` pairs — the hashable,
    deterministic form of the config mapping."""

    id: str = field(default="")

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.config_items))
        object.__setattr__(self, "config_items", ordered)
        if not self.id:
            object.__setattr__(
                self, "id", content_id("rgrp", self.spec_id, ordered)
            )

    @classmethod
    def of(
        cls, spec_id: str, config: Mapping[str, ConfigValue]
    ) -> ReplicationGroup:
        return cls(
            spec_id=spec_id,
            config_items=tuple(sorted(config.items())),
        )


def replication_group_of(result: ExperimentResult) -> ReplicationGroup:
    """The replication family this result belongs to, derived from what was
    run — never from what was observed."""
    return ReplicationGroup.of(result.spec_id, result.config)


def group_replications(
    results: Iterable[ExperimentResult],
) -> dict[ReplicationGroup, tuple[ExperimentResult, ...]]:
    """Bucket results by protocol identity. Contradictory results land in the
    same family whenever they were testing the same thing."""
    groups: dict[ReplicationGroup, list[ExperimentResult]] = {}
    for result in results:
        groups.setdefault(replication_group_of(result), []).append(result)
    return {group: tuple(members) for group, members in groups.items()}
