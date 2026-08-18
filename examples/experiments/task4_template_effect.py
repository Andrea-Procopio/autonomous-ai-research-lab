"""Trusted template: class-separation effect measurement (Task 4 catalog).

One of the planner's catalog options. This family measures how classifier
accuracy responds to the geometric separation of two seeded Gaussian
blobs: the same learner is trained and scored on a close pair and a far
pair of blob centers, and the effect is reported as the accuracy gap. The
metric vocabulary this template can measure is declared in the catalog
entry, not here; only the process contract is encoded:

  reads   ARL_RUN_DIR, ARL_CONFIG, ARL_SEED
  writes  $ARL_RUN_DIR/metrics.json  (flat JSON object of finite numbers)

This file is input to the model, identified by content id and sha256, and
plain ASCII throughout. Running it unmodified fails deliberately: an
untouched starting point must never look like an executed experiment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_metrics(metrics: dict[str, float]) -> None:
    """Write the executor contract's metrics file into the run directory."""
    run_dir = Path(os.environ["ARL_RUN_DIR"])
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )


def main() -> None:
    seed = int(os.environ.get("ARL_SEED", "0"))
    # To be completed by the research engineer: seed all randomness from
    # random.Random(seed), implement the assigned procedure exactly, and
    # finish with one write_metrics call carrying every declared metric.
    # Standard library only; plain ASCII source; no network, no files
    # outside ARL_RUN_DIR.
    raise NotImplementedError(
        f"template not implemented (seed {seed}); the research engineer "
        f"must complete it"
    )


if __name__ == "__main__":
    main()
