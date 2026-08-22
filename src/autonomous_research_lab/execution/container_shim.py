"""Trusted container launcher for one experiment job.

Runs as the job's process under :class:`~autonomous_research_lab.execution.
local.LocalExecutor` (which supplies ``ARL_RUN_DIR`` / ``ARL_CONFIG`` /
``ARL_SEED``) and re-launches the experiment entrypoint inside a disposable
container instead of on the host::

    docker run --rm --network none --pull never
        --cap-drop ALL --security-opt no-new-privileges
        --read-only --tmpfs /tmp
        --memory <m> --memory-swap <m> --pids-limit <n> --cpus <c>
        [--shm-size <v>] [--gpus <n>]
        -v <source>:/arl/src:ro  -v <run_dir>:/arl/run
        [-v <dataset>:/arl/data/<name>:ro ...]
        -e ARL_RUN_DIR=/arl/run -e ARL_CONFIG=/arl/run/config.json
        -e HOME=/tmp [-e ARL_DATA_DIR=/arl/data] [-e ARL_SEED=<seed>]
        <image> python /arl/src/<entrypoint>

Containment properties, stated exactly:

* **no network** — ``--network none``; and ``--pull never``, so even launch
  cannot fetch anything: the pinned image must already be present;
* **no host filesystem** beyond the declared mounts: the validated source
  tree, read-only; the job's run directory, writable; and any staged
  dataset directories, read-only by construction — the shim offers no
  writable variant. No home directory, no credentials, no repository;
  ``HOME`` points at the tmpfs ``/tmp``, because the root filesystem is
  read-only and library caches need somewhere legal to land;
* **no privileges** — all capabilities dropped, no privilege escalation, a
  read-only root filesystem with a tmpfs ``/tmp``;
* **finite** — memory, pids and cpus capped by policy, and a wall-clock
  deadline enforced *here*: on expiry the container is killed by name and
  the shim exits 124, so a stuck experiment cannot outlive its record.

The experiment still speaks the ordinary lab contract (``ARL_*`` in,
``metrics.json`` out) — the executor around this shim runs, records,
hashes and collects exactly as it would any local job. This module is
trusted infrastructure: model-generated code appears only as the mounted
source tree, never as arguments interpreted by a shell.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path

TIMEOUT_EXIT_CODE = 124
LAUNCH_FAILURE_EXIT_CODE = 125

_KILL_WAIT_SECONDS = 30.0


def docker_run_command(
    *,
    image: str,
    source_dir: Path,
    entrypoint: str,
    run_dir: Path,
    memory: str,
    pids_limit: int,
    cpus: float,
    seed: str | None,
    container_name: str,
    gpus: int = 0,
    shm_size: str | None = None,
    data_mounts: tuple[tuple[Path, str], ...] = (),
) -> tuple[str, ...]:
    """The exact ``docker run`` invocation, as pure data — the policy is
    testable without a daemon."""
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--pull",
        "never",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--pids-limit",
        str(pids_limit),
        "--cpus",
        str(cpus),
    ]
    if shm_size is not None:
        command.extend(("--shm-size", shm_size))
    if gpus:
        command.extend(("--gpus", str(gpus)))
    command.extend(
        (
            "-v",
            f"{source_dir}:/arl/src:ro",
            "-v",
            f"{run_dir}:/arl/run",
        )
    )
    for host_dir, name in data_mounts:
        # Read-only by construction: there is no writable variant to ask
        # for, so a job cannot corrupt the dataset every later job reads.
        command.extend(("-v", f"{host_dir}:/arl/data/{name}:ro"))
    command.extend(
        (
            "-e",
            "ARL_RUN_DIR=/arl/run",
            "-e",
            "ARL_CONFIG=/arl/run/config.json",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            # The root filesystem is read-only; libraries that insist on a
            # cache (torch, matplotlib) get the tmpfs, which adds nothing
            # reachable that /tmp did not already offer.
            "-e",
            "HOME=/tmp",
        )
    )
    if data_mounts:
        command.extend(("-e", "ARL_DATA_DIR=/arl/data"))
    if seed is not None:
        command.extend(("-e", f"ARL_SEED={seed}"))
    command.extend((image, "python", f"/arl/src/{entrypoint}"))
    return tuple(command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--pids-limit", required=True, type=int)
    parser.add_argument("--cpus", required=True, type=float)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--shm-size", default=None)
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        metavar="HOST_DIR:NAME",
        help="stage a host directory read-only at /arl/data/NAME",
    )
    args = parser.parse_args(argv)

    data_mounts: list[tuple[Path, str]] = []
    for declared in args.data:
        host, _, name = declared.rpartition(":")
        if not host or not name:
            print(
                f"container shim: --data expects HOST_DIR:NAME, got "
                f"{declared!r}",
                file=sys.stderr,
            )
            return LAUNCH_FAILURE_EXIT_CODE
        host_dir = Path(host)
        if not host_dir.is_dir():
            print(
                f"container shim: dataset directory {host_dir} does not "
                f"exist; stage it before running",
                file=sys.stderr,
            )
            return LAUNCH_FAILURE_EXIT_CODE
        data_mounts.append((host_dir, name))

    run_dir_env = os.environ.get("ARL_RUN_DIR")
    if not run_dir_env:
        print("container shim: ARL_RUN_DIR is not set", file=sys.stderr)
        return LAUNCH_FAILURE_EXIT_CODE
    run_dir = Path(run_dir_env)
    if not (args.source / args.entrypoint).is_file():
        print(
            f"container shim: entrypoint {args.entrypoint} not found under "
            f"{args.source}",
            file=sys.stderr,
        )
        return LAUNCH_FAILURE_EXIT_CODE

    container_name = f"arl-{run_dir.name}"
    command = docker_run_command(
        image=args.image,
        source_dir=args.source,
        entrypoint=args.entrypoint,
        run_dir=run_dir,
        memory=args.memory,
        pids_limit=args.pids_limit,
        cpus=args.cpus,
        seed=os.environ.get("ARL_SEED"),
        container_name=container_name,
        gpus=args.gpus,
        shm_size=args.shm_size,
        data_mounts=tuple(data_mounts),
    )
    try:
        completed = subprocess.run(
            command, check=False, timeout=args.timeout
        )
    except subprocess.TimeoutExpired:
        _kill(container_name)
        print(
            f"container shim: experiment exceeded {args.timeout}s; "
            f"container {container_name} killed",
            file=sys.stderr,
        )
        return TIMEOUT_EXIT_CODE
    except OSError as exc:
        print(f"container shim: could not launch docker: {exc}", file=sys.stderr)
        return LAUNCH_FAILURE_EXIT_CODE
    return completed.returncode


def _kill(container_name: str) -> None:
    """Best-effort cleanup: ``--rm`` removes the container once killed."""
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            ("docker", "kill", container_name),
            check=False,
            capture_output=True,
            timeout=_KILL_WAIT_SECONDS,
        )


if __name__ == "__main__":
    sys.exit(main())
