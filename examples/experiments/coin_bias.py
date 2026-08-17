"""A deliberately trivial experiment, used to exercise the execution contract.

Draws Bernoulli samples from a seeded generator and reports the observed rate.
The science is uninteresting; the point is that the number reaching the lab was
produced by a process that ran, and arrives through ``metrics.json`` rather than
through anybody's reasoning.

Contract with the executor:
  reads  ARL_RUN_DIR, ARL_CONFIG, ARL_SEED
  writes $ARL_RUN_DIR/metrics.json
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

DEFAULT_DRAWS = 4000


def main() -> None:
    run_dir = Path(os.environ["ARL_RUN_DIR"])
    seed = int(os.environ.get("ARL_SEED", "0"))

    config: dict[str, object] = {}
    config_path = os.environ.get("ARL_CONFIG")
    if config_path and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text())

    requested = config.get("n_draws")
    n_draws = requested if isinstance(requested, int) else DEFAULT_DRAWS

    rng = random.Random(seed)
    heads = sum(1 for _ in range(n_draws) if rng.random() < 0.5)
    heads_rate = heads / n_draws

    metrics = {
        "n_draws": n_draws,
        "heads": heads,
        "heads_rate": heads_rate,
        "abs_deviation_from_half": abs(heads_rate - 0.5),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"heads_rate={heads_rate:.4f} over {n_draws} draws (seed {seed})")


if __name__ == "__main__":
    main()
