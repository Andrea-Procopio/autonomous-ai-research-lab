"""Replication semantics: independent evidence, shared protocol identity.

Two invariants together:

* two executions never merge into one record (occurrence identity);
* executions of the same spec under the same configuration form one
  replication family — regardless of what they observed.
"""

from __future__ import annotations

from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.core.replication import (
    ReplicationGroup,
    group_replications,
    replication_group_of,
)
from autonomous_research_lab.core.types import ConfigValue

ENVIRONMENT = Environment(python_version="3.11.0", platform="test")


def run(
    job_id: str,
    *,
    spec_id: str = "exp_1",
    config: dict[str, ConfigValue] | None = None,
    metrics: dict[str, float] | None = None,
    seed: int | None = 7,
) -> ExperimentResult:
    return ExperimentResult(
        spec_id=spec_id,
        job_id=job_id,
        status=ExperimentStatus.COMPLETED,
        command=("python", "run.py"),
        environment=ENVIRONMENT,
        config=config if config is not None else {"n_draws": 4000},
        metrics=metrics if metrics is not None else {"rate": 0.5},
        seed=seed,
    )


def test_identical_protocol_is_one_family() -> None:
    """Same spec, same config: one replication group — the group id is
    content-derived, so it reproduces across runs and branches."""
    first = replication_group_of(run("job_1"))
    second = replication_group_of(run("job_2"))
    assert first == second
    assert first.id == second.id


def test_contradictory_results_stay_in_the_same_family() -> None:
    """Grouping never reads outcomes: a run that disagrees with its siblings
    belongs with them — that disagreement is the finding."""
    agree = run("job_1", metrics={"rate": 0.51})
    disagree = run("job_2", metrics={"rate": 0.97})

    groups = group_replications([agree, disagree])
    ((group, members),) = groups.items()
    assert group == replication_group_of(agree)
    assert members == (agree, disagree)


def test_members_remain_independent_evidence() -> None:
    """A family is a grouping, not a merge: identical runs keep distinct
    result records."""
    first, second = run("job_1"), run("job_2")
    assert first.id != second.id
    assert first.metrics == second.metrics

    ((_, members),) = group_replications([first, second]).items()
    assert len(members) == 2


def test_different_config_is_a_different_family() -> None:
    base = run("job_1", config={"n_draws": 4000})
    scaled = run("job_2", config={"n_draws": 8000})
    assert replication_group_of(base) != replication_group_of(scaled)


def test_different_spec_is_a_different_family() -> None:
    assert replication_group_of(run("job_1", spec_id="exp_1")) != replication_group_of(
        run("job_2", spec_id="exp_2")
    )


def test_seed_does_not_split_a_family() -> None:
    """Different seeds under one protocol are statistical replications of the
    same thing; each member keeps its own seed on the record."""
    assert replication_group_of(run("job_1", seed=7)) == replication_group_of(
        run("job_2", seed=8)
    )


def test_group_identity_is_order_insensitive() -> None:
    forward = ReplicationGroup.of("exp_1", {"a": 1, "b": 2})
    backward = ReplicationGroup.of("exp_1", {"b": 2, "a": 1})
    assert forward.id == backward.id
    assert forward == backward
