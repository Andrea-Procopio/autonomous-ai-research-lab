"""The job-binding seam: trusted code fixes how generated source launches.

The model proposes file *content*; everything about execution — command,
environment, mounts, limits, timeouts, required artifacts — is fixed by a
binding, and for live code the binding is a disposable container whose
policy these tests pin as pure data (no daemon required).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autonomous_research_lab.execution.binding import (
    ContainerBinding,
    HostPythonBinding,
)
from autonomous_research_lab.execution.container_shim import (
    LAUNCH_FAILURE_EXIT_CODE,
    docker_run_command,
)
from autonomous_research_lab.execution.container_shim import (
    main as shim_main,
)


def test_host_binding_fixes_command_env_and_artifacts(tmp_path: Path) -> None:
    binding = HostPythonBinding(timeout_seconds=45.0)
    job = binding.bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={"implementation_id": "impl_1"},
        seed=7,
    )
    assert job.command == (sys.executable, str(tmp_path / "experiment.py"))
    assert job.env == {}
    assert job.seed == 7
    assert job.timeout_seconds == 45.0
    assert job.required_artifacts == ("metrics.json",)
    assert job.config["implementation_id"] == "impl_1"


def test_container_binding_launches_the_trusted_shim(tmp_path: Path) -> None:
    binding = ContainerBinding(
        image="python@sha256:deadbeef",
        docker_host="unix:///tmp/docker.sock",
        memory="512m",
        pids_limit=64,
        cpus=1.5,
        timeout_seconds=120.0,
        launch_margin_seconds=30.0,
    )
    job = binding.bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={},
        seed=3,
    )
    assert job.command[:3] == (
        sys.executable,
        "-m",
        "autonomous_research_lab.execution.container_shim",
    )
    rest = job.command[3:]
    assert rest[0:2] == ("--image", "python@sha256:deadbeef")
    assert "--source" in rest and str(tmp_path) in rest
    assert rest[-2:] == ("--timeout", "120.0")
    assert job.env == {"DOCKER_HOST": "unix:///tmp/docker.sock"}
    assert job.timeout_seconds == 150.0  # shim deadline + launch margin
    assert job.required_artifacts == ("metrics.json",)


def test_container_binding_without_docker_host_passes_no_env(
    tmp_path: Path,
) -> None:
    job = ContainerBinding(image="img").bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={},
        seed=None,
    )
    assert job.env == {}


def test_the_container_policy_is_contained_by_construction() -> None:
    command = docker_run_command(
        image="python@sha256:deadbeef",
        source_dir=Path("/work/src"),
        entrypoint="experiment.py",
        run_dir=Path("/work/run"),
        memory="1g",
        pids_limit=128,
        cpus=2.0,
        seed="7",
        container_name="arl-job_1",
    )
    def follows(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert follows("--network") == "none"
    assert follows("--pull") == "never"
    assert follows("--cap-drop") == "ALL"
    assert follows("--security-opt") == "no-new-privileges"
    assert "--read-only" in command
    assert follows("--tmpfs") == "/tmp"
    assert follows("--memory") == "1g"
    assert follows("--memory-swap") == "1g"
    assert follows("--pids-limit") == "128"
    assert follows("--cpus") == "2.0"
    assert "/work/src:/arl/src:ro" in command  # source read-only
    assert "/work/run:/arl/run" in command  # only the run dir is writable
    assert "ARL_RUN_DIR=/arl/run" in command
    assert "ARL_CONFIG=/arl/run/config.json" in command
    assert "ARL_SEED=7" in command
    # The entrypoint runs from the read-only mount, inside the pinned image.
    assert command[-3:] == (
        "python@sha256:deadbeef",
        "python",
        "/arl/src/experiment.py",
    )


def test_the_loaded_policy_is_contained_by_construction() -> None:
    """The GPU, shared-memory and dataset extensions, pinned the same way:
    every flag they add is enumerated here, and nothing else appears."""
    command = docker_run_command(
        image="python@sha256:deadbeef",
        source_dir=Path("/work/src"),
        entrypoint="experiment.py",
        run_dir=Path("/work/run"),
        memory="8g",
        pids_limit=1024,
        cpus=4.0,
        seed="7",
        container_name="arl-job_1",
        gpus=1,
        shm_size="2g",
        data_mounts=((Path("/data/cifar10"), "cifar10"),),
    )

    def follows(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert follows("--gpus") == "1"
    assert follows("--shm-size") == "2g"
    assert "/data/cifar10:/arl/data/cifar10:ro" in command
    assert "ARL_DATA_DIR=/arl/data" in command
    assert "HOME=/tmp" in command
    # The tail is unchanged: the entrypoint still runs from the read-only
    # mount, inside the pinned image.
    assert command[-3:] == (
        "python@sha256:deadbeef",
        "python",
        "/arl/src/experiment.py",
    )


def test_the_default_policy_emits_nothing_gpu_or_data_shaped() -> None:
    command = docker_run_command(
        image="img",
        source_dir=Path("/s"),
        entrypoint="experiment.py",
        run_dir=Path("/r"),
        memory="1g",
        pids_limit=8,
        cpus=1.0,
        seed=None,
        container_name="arl-x",
    )
    assert "--gpus" not in command
    assert "--shm-size" not in command
    assert "ARL_DATA_DIR=/arl/data" not in command
    assert not any("/arl/data" in part for part in command if ":" in part)
    # HOME is tmpfs-bound unconditionally: the root filesystem is
    # read-only, and library caches need somewhere legal to land.
    assert "HOME=/tmp" in command


def test_data_mounts_are_read_only_by_construction() -> None:
    command = docker_run_command(
        image="img",
        source_dir=Path("/s"),
        entrypoint="experiment.py",
        run_dir=Path("/r"),
        memory="1g",
        pids_limit=8,
        cpus=1.0,
        seed=None,
        container_name="arl-x",
        data_mounts=((Path("/d/a"), "a"), (Path("/d/b"), "b")),
    )
    mounts = [
        command[i + 1]
        for i, part in enumerate(command)
        if part == "-v" and "/arl/data/" in command[i + 1]
    ]
    assert len(mounts) == 2
    assert all(mount.endswith(":ro") for mount in mounts)


def test_gpus_reach_the_job_as_occupancy(tmp_path: Path) -> None:
    job = ContainerBinding(image="img", gpus=2).bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={},
        seed=None,
    )
    assert job.gpu_count == 2
    assert tuple(job.command[-2:]) == ("--gpus", "2")

    host = HostPythonBinding(gpu_count=1).bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={},
        seed=None,
    )
    assert host.gpu_count == 1


def test_a_default_binding_declares_no_new_policy(tmp_path: Path) -> None:
    job = ContainerBinding(image="img").bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={},
        seed=None,
    )
    assert job.gpu_count == 0
    assert "--gpus" not in job.command
    assert "--shm-size" not in job.command
    assert "--data" not in job.command


def test_data_mount_names_must_be_plain_and_unique() -> None:
    for bad in ("", "a/b", ".."):
        with pytest.raises(ValueError, match="plain directory name"):
            ContainerBinding(image="img", data_mounts=(("/d", bad),))
    with pytest.raises(ValueError, match="unique"):
        ContainerBinding(
            image="img", data_mounts=(("/d1", "same"), ("/d2", "same"))
        )


def test_a_binding_passes_data_mounts_to_the_shim(tmp_path: Path) -> None:
    job = ContainerBinding(
        image="img", data_mounts=(("/data/cifar10", "cifar10"),)
    ).bind(
        spec_id="exp_1",
        source_dir=tmp_path,
        entrypoint="experiment.py",
        config={},
        seed=None,
    )
    position = job.command.index("--data")
    assert job.command[position + 1] == "/data/cifar10:cifar10"


def test_seedless_jobs_export_no_seed() -> None:
    command = docker_run_command(
        image="img",
        source_dir=Path("/s"),
        entrypoint="experiment.py",
        run_dir=Path("/r"),
        memory="1g",
        pids_limit=8,
        cpus=1.0,
        seed=None,
        container_name="arl-x",
    )
    assert not any(part.startswith("ARL_SEED=") for part in command)


def test_shim_refuses_to_run_without_a_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARL_RUN_DIR", raising=False)
    code = shim_main(
        [
            "--image", "img", "--source", str(tmp_path),
            "--entrypoint", "experiment.py", "--memory", "1g",
            "--pids-limit", "8", "--cpus", "1", "--timeout", "10",
        ]
    )
    assert code == LAUNCH_FAILURE_EXIT_CODE


def test_shim_refuses_a_missing_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARL_RUN_DIR", str(tmp_path))
    code = shim_main(
        [
            "--image", "img", "--source", str(tmp_path / "nowhere"),
            "--entrypoint", "experiment.py", "--memory", "1g",
            "--pids-limit", "8", "--cpus", "1", "--timeout", "10",
        ]
    )
    assert code == LAUNCH_FAILURE_EXIT_CODE
