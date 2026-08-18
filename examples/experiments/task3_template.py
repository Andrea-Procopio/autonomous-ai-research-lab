"""The fixed implementation template for the Task 3 live vertical slice.

This file is *input to the model*, not an executable experiment: the
model-backed research engineer receives it (identified by content id and
sha256) and must complete it into a full implementation of the assigned
specification. The lab's process contract is already encoded so the model
cannot get it structurally wrong:

  reads   ARL_RUN_DIR, ARL_CONFIG, ARL_SEED
  writes  $ARL_RUN_DIR/metrics.json  (flat JSON object of finite numbers)

Running the template as-is fails deliberately: a template is a starting
point, and an unmodified starting point must never look like an executed
experiment. Everything in this file, and everything an implementation adds
to it, stays plain ASCII.
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
    # ``random.Random(seed)``, generate the dataset the procedure
    # describes, train and evaluate the model, and finish with one
    # ``write_metrics`` call carrying every declared metric.
    raise NotImplementedError(
        f"template not implemented (seed {seed}); the research engineer "
        f"must complete it"
    )


if __name__ == "__main__":
    main()
