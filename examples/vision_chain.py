"""One vision brief through all seven stages, training for real.

The canary walk, grown up: the analysis stages run on scripted
instruments with zero network and zero spend, and the seventh stage
executes genuine training through whatever backend the deployment
profile names —

    export ARL_VISION_PROFILE=/path/to/profile.json
    python -m examples.vision_chain --run-root /tmp/vision
    python -m examples.vision_chain --run-root /tmp/vision --stop-after funding
    python -m examples.vision_chain --run-root /tmp/vision   # continues

With ``--ci`` the trainer is the stdlib stub and no profile, dataset,
torch, or docker is needed: the exact walk the test suite runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autonomous_research_lab.control.controller import Controller, WalkResult
from autonomous_research_lab.control.lab import Lab
from autonomous_research_lab.control.stage import CHAIN_ORDER, StageName
from examples.vision_lab import ci_lab, qualification_lab

CONFIG = Path(__file__).resolve().parent / "vision.json"


def walk(
    root: Path | str,
    *,
    lab: Lab,
    stop_after: StageName | None = None,
) -> WalkResult:
    controller = Controller(Path(root))
    existing = controller.investigations.investigations()
    if existing:
        return controller.resume(
            existing[0].investigation_id, lab=lab, stop_after=stop_after
        )
    payload = json.loads(CONFIG.read_text())
    return controller.run(payload, lab=lab, stop_after=stop_after)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--stop-after", choices=[str(stage) for stage in CHAIN_ORDER]
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="stdlib stub trainer; no profile, dataset, torch, or docker",
    )
    arguments = parser.parse_args()
    stop_after = (
        StageName(arguments.stop_after) if arguments.stop_after else None
    )
    chosen = ci_lab() if arguments.ci else qualification_lab()
    result = walk(arguments.run_root, lab=chosen, stop_after=stop_after)
    print(f"investigation {result.investigation_id}")
    for event in result.events:
        print(f"  {event.stage:<16} {event.status:<10} {event.detail[:70]}")
    print(f"\n{result.outcome}: {result.detail}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
