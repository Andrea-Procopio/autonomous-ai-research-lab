"""The real trainer, smoke-sized: runs only where torch is installed.

CI is torchless by design (the container backend gets torch from the
operator's image), so this module skips itself there. An operator with
``pip install -e ".[vision]"`` and a staged CIFAR-10 gets the template
proven end-to-end at smoke size.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from autonomous_research_lab.core.prediction import Comparator, Prediction
from examples.vision_lab.catalog import catalog_for, fill_slot
from examples.vision_lab.composition import REAL_SLOT
from examples.vision_lab.datasets import DatasetStore

DATASETS_ROOT = os.environ.get("ARL_DATASETS_ROOT", "")

ENCODER_METRIC = (
    "difference in linear probe accuracy: "
    "trained encoder minus randomly initialized encoder"
)


def admitted(metric: str) -> Prediction:
    return Prediction(
        hypothesis_id="hyp_1",
        condition="on held-out images from the evaluation split",
        metric=metric,
        comparator=Comparator.GREATER_THAN,
        threshold=0.0,
        expectation="the trained arm stays above the untrained arm",
    )


@pytest.mark.skipif(
    not DATASETS_ROOT, reason="set ARL_DATASETS_ROOT to a staged CIFAR-10"
)
def test_the_encoder_template_runs_at_smoke_size(tmp_path: Path) -> None:
    store = DatasetStore(DATASETS_ROOT)
    assert store.verify("cifar10") == (), "stage CIFAR-10 first"

    catalog = catalog_for((admitted(ENCODER_METRIC),), trainer="real")
    (entry,) = catalog.entries
    completed = fill_slot(entry.template.source, REAL_SLOT)
    script = tmp_path / "experiment.py"
    script.write_text(completed)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_root": str(store.path_for("cifar10")),
                "smoke": True,
            }
        )
    )

    completed_run = subprocess.run(
        (sys.executable, str(script)),
        env={
            **os.environ,
            "ARL_RUN_DIR": str(run_dir),
            "ARL_CONFIG": str(run_dir / "config.json"),
            "ARL_SEED": "11",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed_run.returncode == 0, completed_run.stderr[-2000:]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert ENCODER_METRIC in metrics
    for key in entry.metrics:
        assert key in metrics, f"missing {key}"
    assert metrics["tiny_subset_overfit_top1"] >= 0.95
