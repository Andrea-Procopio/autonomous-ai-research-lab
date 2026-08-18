"""The generic implementation template for the trajectory campaign.

One template for every campaign task, on purpose: Task 3 proved the slice
with a purpose-built template, and the campaign measures whether the
engineer, the deterministic gates and the verification layer generalize
when the template knows nothing about the experiment. Only the process
contract is encoded:

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
