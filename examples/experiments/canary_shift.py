"""The canary's one experiment: a seeded, invented accuracy contrast.

The science is nothing. Two numbers are drawn from a seeded generator
and their difference is reported under whatever metric name the admitted
prediction asked for, so the pre-registered comparison has something
real to be tested against — a number produced by a process that ran,
arriving through ``metrics.json`` rather than through anybody's
reasoning.

Contract with the executor:
  reads  ARL_RUN_DIR, ARL_CONFIG, ARL_SEED
  writes $ARL_RUN_DIR/metrics.json

``skip_metrics`` in the config makes the run exit without writing them,
which is how the canary produces a failure worth repairing.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

N_PROMPTS = 512


def main() -> None:
    run_dir = Path(os.environ["ARL_RUN_DIR"])
    seed = int(os.environ.get("ARL_SEED", "0"))

    config: dict[str, object] = {}
    config_path = os.environ.get("ARL_CONFIG")
    if config_path and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text())
    metric = config.get("metric")
    if not isinstance(metric, str) or not metric:
        raise SystemExit("the canary experiment needs a metric name in config")

    if config.get("skip_metrics"):
        # The one repairable failure the canary can produce on demand: a
        # process that exits cleanly and writes nothing. The executor
        # records "wrote no metrics.json", the classifier calls that a
        # repairable missing-metrics failure, and the repair strategy
        # reruns the same experiment without this flag. Nothing about the
        # science changes — the point is to have a *failure* that a
        # bounded repair loop can genuinely fix.
        return

    rng = random.Random(seed)
    treatment = 0.70 + rng.uniform(-0.02, 0.02)
    control = 0.63 + rng.uniform(-0.02, 0.02)

    metrics = {
        metric: treatment - control,
        "treatment_accuracy": treatment,
        "control_accuracy": control,
        "n_prompts": N_PROMPTS,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
