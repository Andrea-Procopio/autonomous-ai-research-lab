"""Rerun a preserved implementation at its recorded seed — no model call.

The seed-29 ridge replication of campaign 2026-08-18 generated a valid
implementation (``impl_20b0dd7bbddba420``, source ``src_ff6c61eef7c9aa49``)
whose container launch then died with ``ModuleNotFoundError`` — the
externally caused hidden-``.pth`` condition (see README, Troubleshooting).
The science does not need a new generation: the seed rides in as
``ARL_SEED``, so the preserved bytes run at the recorded seed unchanged.

This script is the rerun vehicle, and it is *incapable* of a model call:
no provider, ledger, or role is imported anywhere in the module. It reads
the preserved implementation record (whose store re-derives the id, so
tampered bytes fail loudly), verifies the source tree against the record's
manifest, proves the preserved bytes implement the pre-registered campaign
spec, and then executes through the exact container policy the campaign
used — preflight first, so the hidden-``.pth`` condition is diagnosed with
its remediation before any launch. Everything it writes lands under a NEW
run root; the original trajectory directories are never write targets.

Run book (the script does not manage the daemon)::

    colima start
    docker image inspect <pinned digest>     # --pull never will not fetch
    python -m examples.rerun_replication --run-root <new dir>
    colima stop

Exit status: 0 only when the rerun result is durably VERIFIED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from autonomous_research_lab.core.experiment import ExperimentResult
from autonomous_research_lab.execution.binding import ContainerBinding
from autonomous_research_lab.execution.local import LocalExecutor
from autonomous_research_lab.runtime.implementation_store import (
    ImplementationRecord,
    ImplementationStore,
)
from autonomous_research_lab.runtime.preflight import (
    PreflightError,
    require_preflight,
)
from autonomous_research_lab.runtime.validation import (
    validate_result,
    verify_artifact_integrity,
)
from autonomous_research_lab.runtime.verification import (
    CheckState,
    ValidityDimension,
    VerificationCheck,
    VerificationReport,
    evaluate_controls,
)
from autonomous_research_lab.runtime.verification_store import (
    FileVerificationStore,
    VerificationRecord,
)
from examples.trajectory_campaign import (
    DEFAULT_IMAGE,
    RIDGE_REPLICATION,
    StructuralMethodology,
)

_POLL_SECONDS = 0.05

_DEFAULT_SOURCE_ROOT = "live_runs/campaign-2026-08-18/ridge-replication"
_DEFAULT_IMPLEMENTATION = "impl_20b0dd7bbddba420"
_DEFAULT_RUN_ROOT = (
    "live_runs/campaign-2026-08-18/ridge-replication-seed29-rerun"
)


def _load_record(
    source_root: Path, implementation_id: str
) -> tuple[ImplementationRecord, Path]:
    """The preserved record and its verified source tree.

    The store re-derives the record id on read, so a tampered record fails
    there; this function additionally re-hashes every manifest file so the
    bytes about to run are provably the bytes that were preserved.
    """
    store = ImplementationStore(source_root / "implementations")
    record = store.get(implementation_id)
    if record is None:
        raise SystemExit(
            f"no implementation record {implementation_id!r} under "
            f"{source_root}"
        )
    tree = store.source_dir(record.source_id)
    for name, expected in record.manifest.items():
        path = tree / name
        if not path.is_file():
            raise SystemExit(f"preserved source file missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"preserved bytes for {name} hash {digest}, but the record "
                f"pinned {expected} — refusing to run unverified source"
            )
    return record, tree


def _original_jobs(source_root: Path, implementation_id: str) -> list[str]:
    """Job ids of the original (failed) executions of this implementation,
    discovered read-only from the preserved run directories."""
    jobs: list[str] = []
    for config_path in sorted(source_root.glob("runs/*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("implementation_id") == implementation_id:
            jobs.append(config_path.parent.name)
    return jobs


def _verification_report(
    result: ExperimentResult, *, execution_passed: bool
) -> VerificationReport:
    """The same four-check report shape the campaign runtime records."""
    task = RIDGE_REPLICATION
    checks: list[VerificationCheck] = [
        VerificationCheck(
            dimension=ValidityDimension.EXECUTION,
            name="deterministic_validation",
            state=CheckState.PASS if execution_passed else CheckState.FAIL,
            detail=(
                "process completed and passed the pre-commit gate "
                "(declared metrics, finite values, seed, artifact "
                "integrity)"
                if execution_passed
                else "deterministic validation failed"
            ),
        )
    ]
    checks.extend(evaluate_controls(task.controls, result.metrics))
    checks.append(
        StructuralMethodology().review(
            task.spec, task.prediction, objective=task.question.text
        )
    )
    checks.append(
        VerificationCheck(
            dimension=ValidityDimension.ANALYSIS,
            name="raw_result_reading",
            state=CheckState.PASS,
            detail=(
                "outcome read by the pre-registered mechanical prediction "
                "check; no downstream aggregation involved"
            ),
        )
    )
    return VerificationReport(checks=tuple(checks))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=_DEFAULT_SOURCE_ROOT)
    parser.add_argument("--implementation-id", default=_DEFAULT_IMPLEMENTATION)
    parser.add_argument("--run-root", default=_DEFAULT_RUN_ROOT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--docker-host",
        default=f"unix://{Path.home()}/.colima/default/docker.sock",
    )
    args = parser.parse_args(argv)

    # Absolute paths throughout: the shim child validates the source tree
    # from a job-private working directory, and docker mounts need
    # absolute paths — a relative --source-root would break both.
    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    record, tree = _load_record(source_root, args.implementation_id)

    task = RIDGE_REPLICATION
    if record.spec_id != task.spec.id:
        raise SystemExit(
            f"the preserved record implements spec {record.spec_id}, not "
            f"the pre-registered ridge replication {task.spec.id}"
        )
    if record.seed is None:
        raise SystemExit("the preserved record carries no seed to rerun")

    originals = _original_jobs(source_root, record.id)
    binding = ContainerBinding(
        image=args.image, docker_host=args.docker_host, timeout_seconds=180.0
    )
    job = binding.bind(
        spec_id=task.spec.id,
        source_dir=tree,
        entrypoint=record.entrypoint,
        config={
            "spec_id": task.spec.id,
            "source_id": record.source_id,
            "implementation_id": record.id,
            "rerun_of_job_id": ",".join(originals) or "unknown",
        },
        seed=record.seed,
    )

    # Preflight before docker: the hidden-.pth condition that killed the
    # original launch is diagnosed here with its remediation, loudly.
    try:
        require_preflight(job, task.spec)
    except PreflightError as error:
        print(f"preflight refused the launch: {error}")
        for check in error.report.failures:
            print(f"  {check.name}: {check.detail}")
        return 1

    executor = LocalExecutor(run_root / "runs")
    job_id = executor.submit(job)
    while not executor.status(job_id).is_terminal:
        time.sleep(_POLL_SECONDS)
    result = executor.collect(job_id)

    validation = validate_result(task.spec, result, prediction=task.prediction)
    integrity = verify_artifact_integrity(result)
    execution_passed = validation.passed and integrity.passed

    report = _verification_report(result, execution_passed=execution_passed)
    verdict = FileVerificationStore(run_root / "verifications").record(
        VerificationRecord(
            result_id=result.id, spec_id=result.spec_id, report=report
        )
    )

    observed = result.metrics.get(task.prediction.metric)
    held = task.prediction.check(observed) if observed is not None else None
    summary: dict[str, object] = {
        "implementation_id": record.id,
        "source_id": record.source_id,
        "source_sha256": dict(record.manifest),
        "spec_id": task.spec.id,
        "seed": record.seed,
        "rerun_of_job_ids": originals,
        "provider_provenance": {
            "request_fingerprint": record.request_fingerprint,
            "response_id": record.response_id,
            "provider_request_id": record.provider_request_id,
            "served_model": record.served_model,
        },
        "new_job_id": job_id,
        "result_id": result.id,
        "result_status": result.status.value,
        "metrics": dict(result.metrics),
        "validation_failures": [c.name for c in validation.failures]
        + ([] if integrity.passed else [integrity.name]),
        "verification": {
            "validity": verdict.validity.value,
            "standing": verdict.standing.value,
        },
        "prediction": {
            "metric": task.prediction.metric,
            "comparator": task.prediction.comparator.value,
            "threshold": task.prediction.threshold,
            "observed": observed,
            "held": held,
        },
        "image": args.image,
        "model_calls_made": 0,
    }
    (run_root / "rerun_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict.validity.value == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
