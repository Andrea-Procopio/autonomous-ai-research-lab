"""Where a job runs, decided by deployment data, invisible to science.

The user's requirement, stated as an invariant: experiments may run on a
laptop CPU, an Apple GPU, a local CUDA card, or (one day) a cluster —
and *nothing scientific may vary with the choice*. The spec, the
predictions, the state, and every gate decision are byte-identical
across backends; what differs is the job's command, its declared GPU
occupancy, its timeout, and the path at which the dataset appears —
execution provenance, recorded where execution is recorded.

An :class:`ExecutionProfile` is the deployment datum (parsed from JSON,
validated loudly at composition time, never written into any record). A
:func:`resolve` call turns it into the existing seams — a
:class:`~autonomous_research_lab.execution.binding.JobBinding` and a
:class:`~autonomous_research_lab.execution.local.LocalExecutor` factory
— plus the one function that knows where a job will see its dataset.
Remote backends are deliberately absent: the ``Executor`` ABC is the
seam they will arrive through, and a profile kind is all they will need
added here.

One doctrine survives every profile: live model-generated code runs in a
container. A host backend refuses to compose with a generating engineer
unless the deployment file says, in so many words,
``allow_generated_code_on_host`` — an operator's recorded decision,
never a default.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from autonomous_research_lab.core.types import ConfigValue
from autonomous_research_lab.execution.binding import (
    ContainerBinding,
    HostPythonBinding,
    JobBinding,
)
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor

from .datasets import DATASET_ID_KEY, DATASET_ROOT_KEY, DatasetStore

CONTAINER_DATA_ROOT = "/arl/data"


class ProfileError(ValueError):
    """A deployment profile that cannot be composed into a backend."""


class Backend(StrEnum):
    HOST_CPU = "host-cpu"
    HOST_MPS = "host-mps"
    HOST_CUDA = "host-cuda"
    CONTAINER_CPU = "container-cpu"
    CONTAINER_CUDA = "container-cuda"

    @property
    def containerized(self) -> bool:
        return self in {Backend.CONTAINER_CPU, Backend.CONTAINER_CUDA}


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """One deployment's answer to "where do jobs run?".

    Data an operator writes, not something any record stores: the run's
    provenance captures what actually executed (command, image, GPU
    occupancy) through the job record, and this object is only the
    recipe."""

    backend: Backend
    datasets_root: Path
    image: str = ""
    """Digest-pinned container image; required for container backends.
    Never hardcoded in the repository — the operator pre-pulled it, the
    operator names it."""

    docker_host: str | None = None
    cpus: float = 4.0
    memory: str = "8g"
    gpu_count: int = 0
    shm_size: str | None = "1g"
    timeout_seconds: float = 3600.0
    allow_generated_code_on_host: bool = False

    def validate(self) -> None:
        """Refuse loudly at composition time, before any stage runs."""
        if self.backend.containerized and "@sha256:" not in self.image:
            raise ProfileError(
                f"backend {self.backend} needs a digest-pinned image "
                f"(got {self.image!r}); pull it first and name it by digest"
            )
        if (
            self.backend in {Backend.CONTAINER_CUDA, Backend.HOST_CUDA}
            and sys.platform != "linux"
        ):
            raise ProfileError(
                f"backend {self.backend} needs a Linux host: there is no "
                f"GPU passthrough under a macOS Docker VM, colima included"
            )
        if self.backend is Backend.HOST_MPS and sys.platform != "darwin":
            raise ProfileError("backend host-mps is an Apple-silicon host")
        if self.timeout_seconds <= 0:
            raise ProfileError("timeout_seconds must be positive")
        if self.gpu_count < 0:
            raise ProfileError("gpu_count cannot be negative")

    @property
    def effective_gpus(self) -> int:
        """MPS occupies the one Apple GPU whatever the profile says; CUDA
        backends occupy what the operator declared; CPU occupies none."""
        if self.backend is Backend.HOST_MPS:
            return 1
        if self.backend in {Backend.HOST_CUDA, Backend.CONTAINER_CUDA}:
            return max(1, self.gpu_count)
        return 0


@dataclass(frozen=True, slots=True)
class ResolvedBackend:
    """A profile, turned into the seams the lab composes with."""

    binding: JobBinding
    executor_factory: Callable[[Path], LocalExecutor]
    dataset_root_for_job: Callable[[str], str]
    """Where a job's process will see the named dataset — a host path for
    host backends, ``/arl/data/<name>`` inside a container."""

    generated_code_allowed: bool
    """Whether a model-completing engineer may be composed with this
    binding. Containers: always. Hosts: only by the operator's explicit
    say-so in the deployment file."""


def resolve(
    profile: ExecutionProfile, *, dataset_names: tuple[str, ...] = ()
) -> ResolvedBackend:
    """The profile's recipe, cooked against the existing seams."""
    profile.validate()
    if profile.backend.containerized:
        binding: JobBinding = ContainerBinding(
            image=profile.image,
            docker_host=profile.docker_host,
            memory=profile.memory,
            pids_limit=1024,
            cpus=profile.cpus,
            timeout_seconds=profile.timeout_seconds,
            launch_margin_seconds=120.0,
            gpus=profile.effective_gpus,
            shm_size=profile.shm_size,
            data_mounts=tuple(
                (str(profile.datasets_root / name), name)
                for name in dataset_names
            ),
        )
        return ResolvedBackend(
            binding=binding,
            executor_factory=lambda root: LocalExecutor(root / "runs"),
            dataset_root_for_job=lambda name: f"{CONTAINER_DATA_ROOT}/{name}",
            generated_code_allowed=True,
        )
    binding = HostPythonBinding(
        timeout_seconds=profile.timeout_seconds,
        gpu_count=profile.effective_gpus,
    )
    return ResolvedBackend(
        binding=binding,
        executor_factory=lambda root: LocalExecutor(root / "runs"),
        dataset_root_for_job=lambda name: str(profile.datasets_root / name),
        generated_code_allowed=profile.allow_generated_code_on_host,
    )


@dataclass(frozen=True, slots=True)
class DatasetBoundBinding:
    """A binding that stamps the job's dataset into its config.

    The engineer names only its own keys (spec, source, implementation);
    which bytes a job may read and where it will find them are trusted
    infrastructure decisions, and the binding is where infrastructure
    already turns source into a job. The id is machine-independent by
    construction; the root is the one value that legitimately differs by
    backend, and it differs *here*, below every scientific record.
    """

    inner: JobBinding
    store: DatasetStore
    dataset_name: str
    dataset_root_for_job: Callable[[str], str]

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
        manifest = self.store.manifest(self.dataset_name)
        stamped: dict[str, ConfigValue] = dict(config)
        stamped[DATASET_ID_KEY] = manifest.id
        stamped[DATASET_ROOT_KEY] = self.dataset_root_for_job(
            self.dataset_name
        )
        return self.inner.bind(
            spec_id=spec_id,
            source_dir=source_dir,
            entrypoint=entrypoint,
            config=stamped,
            seed=seed,
            job_id=job_id,
        )


def profile_from(payload: Mapping[str, object]) -> ExecutionProfile:
    """A profile from parsed JSON, refusing unknown keys the way the run
    config does: a misspelled setting must fail, not silently default."""
    known = {
        "backend",
        "datasets_root",
        "image",
        "docker_host",
        "cpus",
        "memory",
        "gpu_count",
        "shm_size",
        "timeout_seconds",
        "allow_generated_code_on_host",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ProfileError(f"unknown profile key(s): {', '.join(unknown)}")
    if "backend" not in payload or "datasets_root" not in payload:
        raise ProfileError("a profile needs at least backend and datasets_root")
    try:
        backend = Backend(str(payload["backend"]))
    except ValueError as exc:
        raise ProfileError(
            f"unknown backend {payload['backend']!r}; one of "
            f"{', '.join(b.value for b in Backend)}"
        ) from exc
    profile = ExecutionProfile(
        backend=backend,
        datasets_root=Path(str(payload["datasets_root"])),
        image=str(payload.get("image", "")),
        docker_host=(
            str(payload["docker_host"])
            if payload.get("docker_host") is not None
            else None
        ),
        cpus=float(_number(payload, "cpus", 4.0)),
        memory=str(payload.get("memory", "8g")),
        gpu_count=int(_number(payload, "gpu_count", 0)),
        shm_size=(
            str(payload["shm_size"])
            if payload.get("shm_size") is not None
            else None
        ),
        timeout_seconds=float(_number(payload, "timeout_seconds", 3600.0)),
        allow_generated_code_on_host=bool(
            payload.get("allow_generated_code_on_host", False)
        ),
    )
    profile.validate()
    return profile


def _number(payload: Mapping[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{key} must be a number")
    return float(value)
