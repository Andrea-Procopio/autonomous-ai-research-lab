"""Content-addressed artifact storage.

A result names its outputs by absolute path into a run directory, and
the run's own ``manifest.json`` lives in that same directory. Both go
away together. These tests pin what the store keeps, and — more of
them — what it refuses to keep.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.evidence.artifacts import (
    MANIFEST_FILENAME,
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactRefusedError,
    FileArtifactStore,
)

ENVIRONMENT = Environment(python_version="3.11.9", platform="test")


def make_run_dir(
    root: Path,
    *,
    name: str = "job-1",
    artifacts: dict[str, str] | None = None,
    write_manifest: bool = True,
) -> Path:
    """A run directory shaped the way the local executor leaves one."""
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text("ran\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    files = artifacts if artifacts is not None else {"metrics.json": '{"x": 1}'}
    for relative, text in files.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if write_manifest:
        (run_dir / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    relative: hashlib.sha256(
                        (run_dir / relative).read_bytes()
                    ).hexdigest()
                    for relative in files
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return run_dir


def make_result(run_dir: Path, *, job_id: str = "job_1") -> ExperimentResult:
    artifacts = tuple(
        str(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name not in {"stdout.log", "stderr.log", MANIFEST_FILENAME}
    )
    return ExperimentResult(
        spec_id="exp_1",
        job_id=job_id,
        status=ExperimentStatus.COMPLETED,
        command=("python", "experiment.py"),
        environment=ENVIRONMENT,
        metrics={"x": 1.0},
        seed=7,
        artifacts=artifacts,
        logs=(str(run_dir / "stdout.log"), str(run_dir / "stderr.log")),
    )


class TestIngest:
    def test_every_named_file_is_stored_with_its_kind(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")

        manifest = store.ingest(make_result(run_dir))

        by_path = {entry.path: entry for entry in manifest.entries}
        assert set(by_path) == {"metrics.json", "stdout.log", "stderr.log"}
        assert by_path["metrics.json"].kind is ArtifactKind.ARTIFACT
        assert by_path["stdout.log"].kind is ArtifactKind.LOG
        assert by_path["metrics.json"].media_type == "application/json"
        assert by_path["stdout.log"].media_type == "text/plain"
        for entry in manifest.entries:
            assert store.blob_path(entry.digest).is_file()

    def test_the_bytes_survive_the_run_directory(self, tmp_path: Path) -> None:
        """The whole point: delete the run, keep the evidence."""
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")
        manifest = store.ingest(make_result(run_dir))
        payload = (run_dir / "metrics.json").read_bytes()

        for path in sorted(run_dir.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()

        entry = next(e for e in manifest.entries if e.path == "metrics.json")
        assert store.blob_path(entry.digest).read_bytes() == payload

    def test_ingest_is_idempotent(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")
        result = make_result(run_dir)

        first = store.ingest(result)
        second = store.ingest(result)

        assert first == second
        assert len(store.manifests()) == 1

    def test_a_missing_blob_is_restored_by_re_ingest(
        self, tmp_path: Path
    ) -> None:
        """A crash between the blobs and the record leaves a partial
        store; the honest re-run must fill it in rather than trust the
        record it finds."""
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")
        result = make_result(run_dir)
        manifest = store.ingest(result)
        entry = manifest.entries[0]
        store.blob_path(entry.digest).unlink()

        store.ingest(result)

        assert store.blob_path(entry.digest).is_file()

    def test_identical_bytes_are_stored_once(self, tmp_path: Path) -> None:
        first_dir = make_run_dir(tmp_path, name="job-1")
        second_dir = make_run_dir(tmp_path, name="job-2")
        store = FileArtifactStore(tmp_path / "store")

        store.ingest(make_result(first_dir, job_id="job_1"))
        store.ingest(make_result(second_dir, job_id="job_2"))

        blobs = list((tmp_path / "store" / "blobs").rglob("*"))
        stored = [path for path in blobs if path.is_file()]
        # stdout, stderr, metrics.json — the same three bodies twice over.
        assert len(stored) == 3
        assert len(store.manifests()) == 2

    def test_a_result_with_nothing_to_store_gets_an_empty_manifest(
        self, tmp_path: Path
    ) -> None:
        store = FileArtifactStore(tmp_path / "store")
        bare = ExperimentResult(
            spec_id="exp_1",
            job_id="job_1",
            status=ExperimentStatus.FAILED,
            command=("python", "experiment.py"),
            environment=ENVIRONMENT,
            failure_reason="launch failed",
        )

        manifest = store.ingest(bare)

        assert manifest.entries == ()
        assert manifest.total_bytes == 0

    def test_no_scratch_file_is_left_behind(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")
        store.ingest(make_result(run_dir))

        assert list((tmp_path / "store").rglob("*.tmp")) == []


class TestRefusals:
    def test_a_path_outside_the_run_directory_is_refused(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        outsider = tmp_path / "elsewhere.txt"
        outsider.write_text("not this run's output", encoding="utf-8")
        store = FileArtifactStore(tmp_path / "store")
        result = make_result(run_dir)
        smuggled = ExperimentResult(
            spec_id=result.spec_id,
            job_id=result.job_id,
            status=result.status,
            command=result.command,
            environment=result.environment,
            artifacts=(*result.artifacts, str(outsider)),
            logs=result.logs,
        )

        with pytest.raises(ArtifactRefusedError, match="outside its run"):
            store.ingest(smuggled)

        assert store.manifests() == ()
        assert list((tmp_path / "store" / "blobs").rglob("*")) == []

    def test_an_edited_file_is_refused_against_the_runs_own_manifest(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        (run_dir / "metrics.json").write_text('{"x": 999}', encoding="utf-8")
        store = FileArtifactStore(tmp_path / "store")

        with pytest.raises(ArtifactRefusedError, match="post-hoc edit"):
            store.ingest(result)

        assert store.manifests() == ()

    def test_a_file_over_the_ceiling_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path, artifacts={"big.bin": "x" * 4096})
        store = FileArtifactStore(tmp_path / "store", max_blob_bytes=1024)

        with pytest.raises(ArtifactRefusedError, match=r"big\.bin"):
            store.ingest(make_result(run_dir))

    def test_a_vanished_file_is_refused(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        (run_dir / "metrics.json").unlink()
        store = FileArtifactStore(tmp_path / "store")

        with pytest.raises(ArtifactRefusedError, match="not a file"):
            store.ingest(result)

    def test_artifacts_without_logs_cannot_be_confined(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")
        unlogged = ExperimentResult(
            spec_id="exp_1",
            job_id="job_1",
            status=ExperimentStatus.COMPLETED,
            command=("python", "experiment.py"),
            environment=ENVIRONMENT,
            artifacts=(str(run_dir / "metrics.json"),),
        )

        with pytest.raises(ArtifactRefusedError, match="cannot be located"):
            store.ingest(unlogged)

    def test_a_run_without_its_own_manifest_still_stores(
        self, tmp_path: Path
    ) -> None:
        """A foreign or older executor wrote no manifest. There is nothing
        to cross-check against, and refusing would lose the bytes; the
        store records what it hashed."""
        run_dir = make_run_dir(tmp_path, write_manifest=False)
        store = FileArtifactStore(tmp_path / "store")

        manifest = store.ingest(make_result(run_dir))

        assert {entry.path for entry in manifest.entries} == {
            "metrics.json",
            "stdout.log",
            "stderr.log",
        }


class TestRecords:
    def test_a_manifest_reloads_from_a_fresh_store(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        written = FileArtifactStore(tmp_path / "store").ingest(result)

        reloaded = FileArtifactStore(tmp_path / "store").get(result.id)

        assert reloaded == written

    def test_an_unknown_result_has_no_manifest(self, tmp_path: Path) -> None:
        assert FileArtifactStore(tmp_path / "store").get("res_nope") is None

    def test_a_tampered_manifest_fails_to_load(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileArtifactStore(tmp_path / "store")
        store.ingest(result)
        path = tmp_path / "store" / "artifacts" / f"{result.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["entries"][0]["size_bytes"] = 1
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(ArtifactIntegrityError, match="re-derives"):
            store.get(result.id)

    def test_a_different_manifest_for_one_result_is_a_conflict(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        store = FileArtifactStore(tmp_path / "store")
        store.ingest(make_result(run_dir))
        (run_dir / "extra.txt").write_text("more output", encoding="utf-8")

        with pytest.raises(ArtifactConflictError, match="never rewritten"):
            store.ingest(make_result(run_dir))
