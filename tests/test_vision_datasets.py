"""Dataset manifests: content-named, write-once, verified by re-reading."""

from __future__ import annotations

from pathlib import Path

import pytest

from autonomous_research_lab.core.types import ConfigValue
from autonomous_research_lab.runtime.verification import CheckState
from examples.vision_lab.datasets import (
    DATASET_ID_KEY,
    DATASET_ROOT_KEY,
    DatasetIntegrityError,
    DatasetManifest,
    DatasetStaged,
    DatasetStore,
    UnknownDatasetError,
)


def fill(destination: Path) -> None:
    (destination / "batch_1.bin").write_bytes(b"a" * 64)
    (destination / "meta").mkdir()
    (destination / "meta" / "labels.txt").write_text("cat\ndog\n")


def staged(root: Path) -> tuple[DatasetStore, DatasetManifest]:
    store = DatasetStore(root / "data")
    manifest = store.stage(
        "toy", fetch=fill, source_url="https://example.org/toy.tar.gz"
    )
    return store, manifest


class TestStaging:
    def test_staging_hashes_every_file(self, tmp_path: Path) -> None:
        _, manifest = staged(tmp_path)

        assert [path for path, _, _ in manifest.files] == [
            "batch_1.bin",
            "meta/labels.txt",
        ]
        assert all(len(digest) == 64 for _, digest, _ in manifest.files)
        assert manifest.total_bytes == 64 + len("cat\ndog\n")

    def test_staging_twice_fetches_once(self, tmp_path: Path) -> None:
        store, first = staged(tmp_path)
        calls = []

        again = store.stage(
            "toy",
            fetch=lambda _dest: calls.append("fetched"),
            source_url="https://example.org/toy.tar.gz",
        )

        assert calls == []
        assert again == first

    def test_the_id_is_the_bytes_not_the_machine(self, tmp_path: Path) -> None:
        """The same bytes staged under two roots carry one id — which is
        what makes dataset_id safe inside job config on any backend."""
        _, here = staged(tmp_path / "a")
        _, there = staged(tmp_path / "b")

        assert here.id == there.id

    def test_an_unknown_dataset_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownDatasetError):
            DatasetStore(tmp_path).manifest("nowhere")

    def test_a_tampered_manifest_refuses_to_load(self, tmp_path: Path) -> None:
        store, manifest = staged(tmp_path)
        path = store.root / "manifests" / "toy.json"
        doctored = path.read_text().replace(
            manifest.files[0][1], "0" * 64
        )
        path.write_text(doctored)

        with pytest.raises(DatasetIntegrityError):
            store.manifest("toy")


class TestVerification:
    def test_intact_files_verify(self, tmp_path: Path) -> None:
        store, _ = staged(tmp_path)
        assert store.verify("toy") == ()

    def test_altered_bytes_are_named(self, tmp_path: Path) -> None:
        store, _ = staged(tmp_path)
        (store.path_for("toy") / "batch_1.bin").write_bytes(b"b" * 64)

        (problem,) = store.verify("toy")

        assert "batch_1.bin" in problem and "no longer match" in problem

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        store, _ = staged(tmp_path)
        (store.path_for("toy") / "batch_1.bin").unlink()

        (problem,) = store.verify("toy")

        assert problem == "batch_1.bin: missing"

    def test_shallow_verification_settles_for_size(self, tmp_path: Path) -> None:
        store, _ = staged(tmp_path)
        (store.path_for("toy") / "batch_1.bin").write_bytes(b"b" * 64)

        assert store.verify("toy", deep=False) == ()


class Job:
    """The JobLike slice, as a preflight check sees it."""

    def __init__(self, config: dict[str, ConfigValue]):
        self.command: tuple[str, ...] = ("python", "x.py")
        self.config: dict[str, ConfigValue] = config
        self.seed: int | None = None
        self.working_dir: str | None = None
        self.required_artifacts: tuple[str, ...] = ()


class TestPreflight:
    def test_a_job_without_a_dataset_is_not_applicable(
        self, tmp_path: Path
    ) -> None:
        store, _ = staged(tmp_path)
        check = DatasetStaged(store).check(Job({}), None)
        assert check.state is CheckState.NOT_APPLICABLE

    def test_a_verified_dataset_passes(self, tmp_path: Path) -> None:
        store, manifest = staged(tmp_path)
        job = Job({DATASET_ID_KEY: manifest.id, DATASET_ROOT_KEY: "/arl/data/toy"})

        check = DatasetStaged(store).check(job, None)

        assert check.state is CheckState.PASS

    def test_an_unstaged_id_fails(self, tmp_path: Path) -> None:
        store, _ = staged(tmp_path)
        check = DatasetStaged(store).check(
            Job({DATASET_ID_KEY: "dset_0000000000000000"}), None
        )
        assert check.state is CheckState.FAIL
        assert "no staged dataset" in check.detail

    def test_altered_bytes_fail_before_launch(self, tmp_path: Path) -> None:
        store, manifest = staged(tmp_path)
        (store.path_for("toy") / "batch_1.bin").write_bytes(b"b" * 64)

        check = DatasetStaged(store).check(
            Job({DATASET_ID_KEY: manifest.id}), None
        )

        assert check.state is CheckState.FAIL
        assert "no longer verifies" in check.detail
