"""Job binding: how validated implementation source becomes an ExperimentJob.

The model-backed engineer validates and preserves generated source; *how*
that source is launched is a trusted infrastructure decision the model never
touches. A :class:`JobBinding` makes that decision one small, swappable
object: given a source tree, an entrypoint, config and a seed, it returns
the :class:`~autonomous_research_lab.execution.executor.ExperimentJob` to
submit. The command, environment, timeout and required artifacts are fixed
by the binding — generated code chooses none of them.

Two bindings, one contract:

* :class:`HostPythonBinding` runs the entrypoint directly with the host
  interpreter. **For trusted fixture source only** (deterministic tests):
  ``LocalExecutor`` provides recovery isolation, not a security sandbox,
  and live model-generated code must not run on the host.
* :class:`ContainerBinding` runs the entrypoint inside a disposable
  container via the :mod:`~autonomous_research_lab.execution.container_shim`
  launcher: network disabled, only the source tree (read-only) and the run
  directory mounted, capabilities dropped, finite memory/pids/cpu/time, a
  pinned preinstalled image, and no runtime pulls. The container's Linux VM
  boundary is what makes live generated code safe to execute at all.

Either way the process obeys the existing lab contract — ``ARL_RUN_DIR``,
``ARL_CONFIG``, ``ARL_SEED`` in; ``metrics.json`` out — and the existing
executor runs, records and collects it unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..core.types import ConfigValue
from .executor import ExperimentJob

METRICS_ARTIFACT = "metrics.json"

_SHIM_MODULE = "autonomous_research_lab.execution.container_shim"


class JobBinding(Protocol):
    """Turn one validated source tree into one runnable job.

    ``job_id`` lets the caller name the job before it exists — see
    :func:`~.executor.derive_job_id`. Empty means the job mints its own,
    which is the right answer wherever nobody is going to have to find
    the job again after a crash.
    """

    def bind(
        self,
        *,
        spec_id: str,
        source_dir: Path,
        entrypoint: str,
        config: Mapping[str, ConfigValue],
        seed: int | None,
        job_id: str = "",
    ) -> ExperimentJob: ...


@dataclass(frozen=True, slots=True)
class HostPythonBinding:
    """Run the entrypoint with the host interpreter — trusted fixture
    source only, never live model output (see module docstring)."""

    timeout_seconds: float = 120.0

    def bind(
        self,
        *,
        spec_id: str,
        source_dir: Path,
        entrypoint: str,
        config: Mapping[str, ConfigValue],
        seed: int | None,
        job_id: str = "",
    ) -> ExperimentJob:
        return ExperimentJob(
            spec_id=spec_id,
            command=(sys.executable, str(source_dir / entrypoint)),
            config=config,
            seed=seed,
            timeout_seconds=self.timeout_seconds,
            required_artifacts=(METRICS_ARTIFACT,),
            id=job_id,
        )


@dataclass(frozen=True, slots=True)
class ContainerBinding:
    """Run the entrypoint in a disposable, network-less container.

    The job's command launches the trusted shim with this policy spelled
    out as arguments; the shim reads the run directory from the executor's
    ``ARL_*`` environment at run time (the run directory does not exist
    when the job is constructed) and drives ``docker run``. The image is
    expected to be present already — the shim never pulls, so a live
    experiment cannot reach the network even at launch time.
    """

    image: str
    """The pinned container image (prefer a digest reference, so the
    environment that ran is the environment recorded)."""

    docker_host: str | None = None
    """Explicit ``DOCKER_HOST`` for the shim, passed via ``job.env``. The
    executor gives children an allowlisted environment and a job-private
    HOME, so a CLI context stored under the real ``~/.docker`` is invisible
    by design — the socket must be named here explicitly."""

    memory: str = "1g"
    pids_limit: int = 128
    cpus: float = 2.0
    timeout_seconds: float = 300.0
    """The in-container deadline the shim enforces (it kills the container
    on expiry). The job's own timeout adds ``launch_margin_seconds`` on
    top, so the shim — not a process-group kill that could strand a
    container — is what normally fires first."""

    launch_margin_seconds: float = 60.0

    def bind(
        self,
        *,
        spec_id: str,
        source_dir: Path,
        entrypoint: str,
        config: Mapping[str, ConfigValue],
        seed: int | None,
        job_id: str = "",
    ) -> ExperimentJob:
        env = (
            {"DOCKER_HOST": self.docker_host}
            if self.docker_host is not None
            else {}
        )
        return ExperimentJob(
            spec_id=spec_id,
            command=(
                sys.executable,
                "-m",
                _SHIM_MODULE,
                "--image",
                self.image,
                "--source",
                str(source_dir),
                "--entrypoint",
                entrypoint,
                "--memory",
                self.memory,
                "--pids-limit",
                str(self.pids_limit),
                "--cpus",
                str(self.cpus),
                "--timeout",
                str(self.timeout_seconds),
            ),
            config=config,
            env=env,
            seed=seed,
            timeout_seconds=self.timeout_seconds + self.launch_margin_seconds,
            required_artifacts=(METRICS_ARTIFACT,),
            id=job_id,
        )
