"""The trajectory campaign's task definitions and driver, deterministically.

The campaign itself runs live; what tests pin is everything the live run
depends on being right in advance: every task is structurally sound
pre-registered science (declared prediction metric, stated baseline,
instrument control that reads a declared metric), and the driver walks a
task from initial state to a natural director stop, producing measurable
records — proven here with the fake provider and a fixture source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.execution.binding import HostPythonBinding
from autonomous_research_lab.runtime.providers import FakeModelProvider
from autonomous_research_lab.runtime.verification import CheckState
from examples.trajectory_campaign import (
    CAMPAIGN_TASKS,
    SEPARABLE_KNN,
    CampaignTask,
    StructuralMethodology,
    measure,
    run_task,
)

KNN_FIXTURE_SOURCE = """\
import json
import os
from pathlib import Path

metrics = {
    "test_accuracy": 0.91,
    "majority_baseline_accuracy": 0.5,
    "tiny_subset_accuracy": 1.0,
    "n_train": 300,
    "n_test": 200,
}
run_dir = Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps(metrics))
"""


def test_campaign_slugs_are_unique() -> None:
    slugs = [task.slug for task in CAMPAIGN_TASKS]
    assert len(slugs) == len(set(slugs))
    assert len(CAMPAIGN_TASKS) == 4


@pytest.mark.parametrize("task", CAMPAIGN_TASKS, ids=lambda t: t.slug)
def test_every_task_is_structurally_sound_science(task: CampaignTask) -> None:
    # The pre-registered chain is internally consistent.
    assert task.prediction.hypothesis_id == task.hypothesis.id
    assert task.spec.prediction_id == task.prediction.id
    assert task.prediction.metric in task.spec.metrics
    assert task.spec.seeds
    assert task.spec.baselines
    assert task.spec.controls
    # Every instrument control reads a metric the run must report.
    assert task.controls
    for control in task.controls:
        assert control.metric in task.spec.metrics
    # And the campaign's structural methodology review agrees.
    check = StructuralMethodology().review(
        task.spec, task.prediction, objective=task.spec.objective
    )
    assert check.state is CheckState.PASS


def test_the_campaign_brief_is_covered() -> None:
    """One baseline-anchored positive, one genuine negative, one
    replication family, one ablation — pinned so the campaign cannot
    quietly lose a requirement."""
    by_slug = {task.slug: task for task in CAMPAIGN_TASKS}
    assert "majority_baseline_accuracy" in by_slug["separable-knn"].spec.metrics
    negative = by_slug["xor-perceptron"]
    assert "expected NOT to hold" in negative.prediction.expectation
    assert len(by_slug["ridge-replication"].spec.seeds) == 3
    assert "ablation_gain" in by_slug["scaling-ablation"].spec.metrics


def test_the_driver_walks_a_task_to_a_natural_stop(tmp_path: Path) -> None:
    reply = json.dumps(
        {
            "files": [
                {"path": "experiment.py", "content": KNN_FIXTURE_SOURCE}
            ],
            "rationale": "fixture kNN implementation",
        }
    )
    run = run_task(
        SEPARABLE_KNN,
        tmp_path / "separable-knn",
        provider=FakeModelProvider((reply,)),
        model="test-model",
        binding=HostPythonBinding(timeout_seconds=60.0),
    )
    # The director stopped on its own: no open work remained.
    assert "no open scientific work" in run.stopped_by

    summary = measure(run)
    assert summary["engineer_invocations"] == 1
    assert summary["results_completed"] == 1
    assert summary["results_failed"] == 0
    assert summary["implementation_success_rate"] == 1.0
    assert summary["generation_repairs"] == 0
    assert summary["admissible_results"] == 1
    assert summary["admissibility_rate"] == 1.0
    assert summary["prediction_tests"] == ["consistent"]
    assert summary["provider_calls"] == 1
    assert summary["claims"] == 1
    assessments = summary["assessments"]
    assert isinstance(assessments, int) and assessments >= 1
    verification = summary["verification"]
    assert isinstance(verification, dict)
    assert set(verification.values()) == {"verified"}
    # The record survives on disk alongside the trajectory and metrics.
    assert (tmp_path / "separable-knn" / "metrics.jsonl").exists()
    assert (tmp_path / "separable-knn" / "trajectory.jsonl").exists()
    assert any((tmp_path / "separable-knn" / "verifications").iterdir())
