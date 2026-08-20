"""Verify a run root from cold.

Point it at a directory a run wrote and it re-checks every durable claim
in it: state snapshots re-hash to their own filenames, result and
evidence payloads survive their digests, every state's references
resolve, every stored artifact still hashes to what its manifest says,
the evidence chain holds, and — for a funded run — the budget ledger
replays.

Nothing here writes. The process that produced the run is gone by the
time this one starts, which is the whole claim being tested::

    python examples/verify_run.py --root <run_root>

Exits 0 when the run is intact and 1 when anything is wrong, so it can
sit in a script. Every issue is printed, not just the first.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from autonomous_research_lab.program.integrity import (
    IntegrityReport,
    verify_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="the run root to verify",
    )
    parser.add_argument(
        "--program-root",
        type=Path,
        default=None,
        help=(
            "where the funded run's records live, if not <root>/program; "
            "omit it for a run that was never funded"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if not root.is_dir():
        print(f"FATAL: {root} is not a directory")
        return 1

    report = verify_run(root, program_root=arguments.program_root)
    _print(report)
    return 0 if report.ok else 1


def _print(report: IntegrityReport) -> None:
    print(f"-- {report.root} --")
    print(f"state snapshots    {report.states_checked}")
    print(f"results            {report.results_checked}")
    print(f"evidence records   {report.evidence_checked}")
    print(f"artifact blobs     {report.blobs_checked}")
    print()
    if report.ok:
        print("intact: every durable claim in this run still holds.")
        print()
        return
    counts = Counter(issue.kind for issue in report.issues)
    print(f"-- {len(report.issues)} issue(s) --")
    for kind, count in sorted(counts.items()):
        print(f"{count:>4}  {kind}")
    print()
    for issue in report.issues:
        print(f"[{issue.kind}] {issue.subject_id}")
        print(f"    {issue.detail}")
    print()
    print("A verified run is one whose records survived; it is not a run")
    print("whose science is right. That is a different question, and the")
    print("assessments answer it.")
    print()


if __name__ == "__main__":
    sys.exit(main())
