"""The canary: one synthetic brief, all seven stages, no outside world.

    brief -> mapping -> ideation -> prior art -> selection
          -> admission -> funding -> experimentation

Every instrument is a fixture (see :mod:`examples.canary_lab`): a
literature provider answering from an invented corpus, a model answering
by computing a reply from the request, and a runtime whose roles and
trusted template are ordinary Python. Nothing here reaches a network, a
clock, or a model, so the whole chain runs in about half a second and
gives the same answer every time.

The research is invented and means nothing. What the run demonstrates is
the machinery: that one config with no ids in it carries a topic from
nothing to a funded run that executes real experiments through the
ordinary executor, records real evidence, bills a real ledger, and
verifies from cold — and that stopping it at any stage boundary and
resuming loses nothing and repeats nothing.

Run with::

    python -m examples.canary_chain --run-root /tmp/canary
    python -m examples.canary_chain --run-root /tmp/canary --stop-after selection
    python -m examples.canary_chain --run-root /tmp/canary   # continues

The second and third commands are the interesting pair: the walk stops
where it was told, and a later process picks it up from the durable
record with no memory of the first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autonomous_research_lab.control.config import load_config
from autonomous_research_lab.control.controller import Controller, WalkResult
from autonomous_research_lab.control.stage import (
    CHAIN_ORDER,
    StageName,
    StageStatus,
)
from autonomous_research_lab.program.integrity import IntegrityReport, verify_run
from examples.canary_lab import lab

CONFIG = Path(__file__).resolve().parent / "canary.json"


def walk(
    root: Path | str, *, stop_after: StageName | None = None
) -> WalkResult:
    """Start the canary under ``root``, or continue the one already there.

    Continuing is the interesting half: nothing is carried over from the
    process that stopped, so everything the walk needs to know it reads
    back from the log.
    """
    controller = Controller(Path(root))
    existing = controller.investigations.investigations()
    if existing:
        return controller.resume(
            existing[0].investigation_id, lab=lab(), stop_after=stop_after
        )
    _, payload = load_config(CONFIG)
    return controller.run(payload, lab=lab(), stop_after=stop_after)


def verify(root: Path | str) -> IntegrityReport:
    root = Path(root)
    return verify_run(root, program_root=root / "program")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, required=True, help="the run directory"
    )
    parser.add_argument(
        "--stop-after",
        choices=[str(stage) for stage in CHAIN_ORDER],
        help="halt this walk after that stage; run again to continue",
    )
    arguments = parser.parse_args()
    root = arguments.run_root.resolve()
    stop_after = (
        StageName(arguments.stop_after) if arguments.stop_after else None
    )

    result = walk(root, stop_after=stop_after)
    print(f"investigation {result.investigation_id}")
    print(f"root          {root}")
    print()
    for event in result.events:
        if event.status is StageStatus.RUNNING:
            continue
        print(
            f"  {event.stage:<16} {event.status:<10} "
            f"{event.spend.model_calls:>3} call(s)  {event.detail[:52]}"
        )
    print()
    print(f"{result.outcome}: {result.detail}")

    report = verify(root)
    print()
    print(
        f"verify: states {report.states_checked}, results "
        f"{report.results_checked}, evidence {report.evidence_checked}, "
        f"blobs {report.blobs_checked}"
    )
    for issue in report.issues:
        print(f"  {issue.kind}: {issue.subject_id}: {issue.detail}")
    if not report.ok:
        print("FATAL: the run does not verify.")
        return 1
    print("intact")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
