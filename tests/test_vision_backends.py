"""Backend profiles: deployment data resolving to seams, leaking nothing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from examples.vision_lab.backends import (
    Backend,
    DatasetBoundBinding,
    ExecutionProfile,
    ProfileError,
    profile_from,
    resolve,
)
from examples.vision_lab.datasets import (
    DATASET_ID_KEY,
    DATASET_ROOT_KEY,
    DatasetStore,
)

IMAGE = "pytorch/pytorch@sha256:" + "0" * 64


def container_profile(tmp_path: Path, **overrides: object) -> ExecutionProfile:
    defaults: dict[str, object] = {
        "backend": Backend.CONTAINER_CPU,
        "datasets_root": tmp_path / "data",
        "image": IMAGE,
    }
    return ExecutionProfile(**(defaults | overrides))  # type: ignore[arg-type]


class TestValidation:
    def test_a_container_backend_needs_a_pinned_image(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ProfileError, match="digest-pinned"):
            container_profile(tmp_path, image="pytorch/pytorch").validate()

    def test_cuda_needs_linux(self, tmp_path: Path) -> None:
        """If/else on the platform check, deliberately: mypy prunes the
        branch for the other platform silently, whereas a guard ending in
        ``pytest.skip`` leaves provably-unreachable code on Linux and
        ``warn_unreachable`` refuses it."""
        profile = container_profile(
            tmp_path, backend=Backend.CONTAINER_CUDA, gpu_count=1
        )
        if sys.platform == "linux":
            profile.validate()  # genuinely valid where CUDA can exist
        else:
            with pytest.raises(ProfileError, match="Linux host"):
                profile.validate()

    def test_mps_needs_darwin(self, tmp_path: Path) -> None:
        profile = ExecutionProfile(
            backend=Backend.HOST_MPS, datasets_root=tmp_path
        )
        if sys.platform == "darwin":
            profile.validate()
            assert profile.effective_gpus == 1
        else:
            with pytest.raises(ProfileError, match="Apple"):
                profile.validate()

    def test_unknown_profile_keys_refuse(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileError, match="unknown profile key"):
            profile_from(
                {
                    "backend": "host-cpu",
                    "datasets_root": str(tmp_path),
                    "tiemout_seconds": 60,
                }
            )

    def test_a_parsed_profile_round_trips(self, tmp_path: Path) -> None:
        profile = profile_from(
            {
                "backend": "container-cpu",
                "datasets_root": str(tmp_path),
                "image": IMAGE,
                "memory": "2g",
                "timeout_seconds": 600,
            }
        )
        assert profile.backend is Backend.CONTAINER_CPU
        assert profile.memory == "2g"
        assert profile.timeout_seconds == 600.0


class TestResolution:
    def test_a_container_profile_mounts_its_datasets(
        self, tmp_path: Path
    ) -> None:
        resolved = resolve(
            container_profile(tmp_path), dataset_names=("cifar10",)
        )
        job = resolved.binding.bind(
            spec_id="exp_1",
            source_dir=tmp_path,
            entrypoint="experiment.py",
            config={},
            seed=1,
        )
        assert "--data" in job.command
        assert resolved.dataset_root_for_job("cifar10") == "/arl/data/cifar10"
        assert resolved.generated_code_allowed

    def test_a_host_profile_uses_host_paths(self, tmp_path: Path) -> None:
        resolved = resolve(
            ExecutionProfile(
                backend=Backend.HOST_CPU, datasets_root=tmp_path / "data"
            )
        )
        assert resolved.dataset_root_for_job("cifar10") == str(
            tmp_path / "data" / "cifar10"
        )
        assert not resolved.generated_code_allowed

    def test_host_generated_code_is_an_explicit_operator_decision(
        self, tmp_path: Path
    ) -> None:
        resolved = resolve(
            ExecutionProfile(
                backend=Backend.HOST_CPU,
                datasets_root=tmp_path,
                allow_generated_code_on_host=True,
            )
        )
        assert resolved.generated_code_allowed


def fill(destination: Path) -> None:
    (destination / "data.bin").write_bytes(b"x" * 16)


class TestNothingScientificVaries:
    def test_the_same_bind_differs_only_in_execution(
        self, tmp_path: Path
    ) -> None:
        """One (spec, source, config, seed) bound through a host and a
        container backend: identical scientific identity, different
        command — provenance doing its job, not leakage."""
        store = DatasetStore(tmp_path / "data")
        store.stage("toy", fetch=fill, source_url="https://example.org/t")

        def bound(profile: ExecutionProfile) -> object:
            resolved = resolve(profile, dataset_names=("toy",))
            return DatasetBoundBinding(
                inner=resolved.binding,
                store=store,
                dataset_name="toy",
                dataset_root_for_job=resolved.dataset_root_for_job,
            ).bind(
                spec_id="exp_1",
                source_dir=tmp_path / "src",
                entrypoint="experiment.py",
                config={"spec_id": "exp_1"},
                seed=7,
                job_id="job_1",
            )

        host = bound(
            ExecutionProfile(
                backend=Backend.HOST_CPU, datasets_root=tmp_path / "data"
            )
        )
        container = bound(container_profile(tmp_path))

        for job in (host, container):
            assert job.spec_id == "exp_1"  # type: ignore[attr-defined]
            assert job.seed == 7  # type: ignore[attr-defined]
            assert job.id == "job_1"  # type: ignore[attr-defined]
            assert job.required_artifacts == ("metrics.json",)  # type: ignore[attr-defined]
        # The dataset id is the bytes, so it cannot vary by backend...
        assert host.config[DATASET_ID_KEY] == container.config[DATASET_ID_KEY]  # type: ignore[attr-defined]
        # ...and the root is the one value that legitimately does.
        assert host.config[DATASET_ROOT_KEY] != container.config[DATASET_ROOT_KEY]  # type: ignore[attr-defined]
        assert container.config[DATASET_ROOT_KEY] == "/arl/data/toy"  # type: ignore[attr-defined]
        assert host.command != container.command  # type: ignore[attr-defined]
