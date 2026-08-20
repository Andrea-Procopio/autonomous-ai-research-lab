"""The durable evidence store.

Two things are pinned here that the in-memory store cannot have. The
ordering — bytes, then the fact — because a recorded result whose
outputs are gone is the gap this store exists to close. And the payload
digest, because the domain ids of results and evidence deliberately do
not cover their own content, so recomputing an id proves nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autonomous_research_lab.core.budget import ResourceCost
from autonomous_research_lab.core.evidence import Evidence, EvidenceKind
from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentStatus,
)
from autonomous_research_lab.evidence.file_store import (
    EvidenceIntegrityError,
    FileEvidenceStore,
)
from autonomous_research_lab.evidence.store import (
    EvidenceConflictError,
    InMemoryEvidenceStore,
    UnknownRecordError,
)

MANIFEST_FILENAME = "manifest.json"


def make_run_dir(root: Path, *, name: str = "job-1") -> Path:
    """A run directory shaped the way the local executor leaves one."""
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text("ran\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"x": 1}', encoding="utf-8")
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "metrics.json": hashlib.sha256(
                    (run_dir / "metrics.json").read_bytes()
                ).hexdigest()
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def make_result(run_dir: Path, *, job_id: str = "job_1") -> ExperimentResult:
    return ExperimentResult(
        spec_id="exp_1",
        job_id=job_id,
        status=ExperimentStatus.COMPLETED,
        command=("python", "experiment.py"),
        environment=Environment(python_version="3.11.9", platform="test"),
        metrics={"x": 1.0},
        seed=7,
        artifacts=(str(run_dir / "metrics.json"),),
        logs=(str(run_dir / "stdout.log"), str(run_dir / "stderr.log")),
    )


def make_evidence(result: ExperimentResult, **overrides: object) -> Evidence:
    values: dict[str, object] = {
        "result_id": result.id,
        "spec_id": result.spec_id,
        "kind": EvidenceKind.MEASUREMENT,
        "observation": "x = 1.0 over one run, seed 7",
        "metrics": {"x": 1.0},
    }
    values.update(overrides)
    return Evidence(**values)  # type: ignore[arg-type]


class TestRoundTrip:
    def test_a_result_reloads_field_for_field(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = ExperimentResult(
            spec_id="exp_1",
            job_id="job_1",
            status=ExperimentStatus.COMPLETED,
            command=("python", "experiment.py", "--fast"),
            environment=Environment(
                python_version="3.11.9",
                platform="darwin",
                git_commit="abc123",
                git_dirty=True,
            ),
            metrics={"heads_rate": 0.503},
            config={"n": 4000, "label": "a", "ratio": 0.5, "flag": True,
                    "absent": None},
            seed=7,
            artifacts=(str(run_dir / "metrics.json"),),
            logs=(str(run_dir / "stdout.log"), str(run_dir / "stderr.log")),
            runtime_seconds=1.25,
            cost=ResourceCost(wall_clock_seconds=1.25, model_tokens=10),
            exit_code=0,
        )
        FileEvidenceStore(tmp_path / "store").record_result(result)

        reloaded = FileEvidenceStore(tmp_path / "store").get_result(result.id)

        assert reloaded == result

    def test_a_failed_result_keeps_its_failure(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = ExperimentResult(
            spec_id="exp_1",
            job_id="job_2",
            status=ExperimentStatus.FAILED,
            command=("python", "experiment.py"),
            environment=Environment(python_version="3.11.9", platform="test"),
            logs=(str(run_dir / "stdout.log"), str(run_dir / "stderr.log")),
            exit_code=1,
            failure_reason="process exited 1",
            seed=None,
        )
        store = FileEvidenceStore(tmp_path / "store")
        store.record_result(result)

        reloaded = FileEvidenceStore(tmp_path / "store").get_result(result.id)

        assert reloaded.failure_reason == "process exited 1"
        assert reloaded.seed is None
        assert not reloaded.succeeded

    def test_evidence_reloads_with_its_metrics(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileEvidenceStore(tmp_path / "store")
        store.record_result(result)
        evidence = store.record_evidence(make_evidence(result))

        reloaded = FileEvidenceStore(tmp_path / "store").get_evidence(evidence.id)

        assert reloaded == evidence
        assert reloaded.metrics == {"x": 1.0}

    def test_listing_reads_everything_back(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileEvidenceStore(tmp_path / "store")
        store.record_result(result)
        store.record_evidence(make_evidence(result))

        fresh = FileEvidenceStore(tmp_path / "store")

        assert [r.id for r in fresh.results()] == [result.id]
        assert len(fresh.evidence()) == 1


class TestTheSameContractAsMemory:
    def test_re_recording_identical_content_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileEvidenceStore(tmp_path / "store")

        assert store.record_result(result) == store.record_result(result)
        assert len(store.results()) == 1

    def test_different_content_under_one_id_raises(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileEvidenceStore(tmp_path / "store")
        store.record_result(result)
        altered = ExperimentResult(
            spec_id=result.spec_id,
            job_id=result.job_id,  # same job, so the same id
            status=result.status,
            command=result.command,
            environment=result.environment,
            metrics={"x": 999.0},
            artifacts=result.artifacts,
            logs=result.logs,
        )
        assert altered.id == result.id

        with pytest.raises(EvidenceConflictError, match="different content"):
            store.record_result(altered)

    def test_evidence_for_an_unrecorded_result_raises(
        self, tmp_path: Path
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        store = FileEvidenceStore(tmp_path / "store")

        with pytest.raises(UnknownRecordError, match="unrecorded result"):
            store.record_evidence(make_evidence(make_result(run_dir)))

    def test_unknown_ids_raise(self, tmp_path: Path) -> None:
        store = FileEvidenceStore(tmp_path / "store")
        with pytest.raises(UnknownRecordError):
            store.get_result("res_nope")
        with pytest.raises(UnknownRecordError):
            store.get_evidence("ev_nope")

    def test_it_behaves_like_the_in_memory_store(self, tmp_path: Path) -> None:
        """The two implementations are interchangeable behind the
        protocol, which is what lets the file store be a drop-in."""
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        evidence = make_evidence(result)
        memory = InMemoryEvidenceStore()
        disk = FileEvidenceStore(tmp_path / "store")

        for store in (memory, disk):
            store.record_result(result)
            store.record_evidence(evidence)

        assert memory.get_result(result.id) == disk.get_result(result.id)
        assert memory.get_evidence(evidence.id) == disk.get_evidence(evidence.id)


class TestOrdering:
    def test_the_bytes_are_stored_before_the_fact(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileEvidenceStore(tmp_path / "store")

        store.record_result(result)

        manifest = store.artifacts.get(result.id)
        assert manifest is not None
        for entry in manifest.entries:
            assert store.artifacts.blob_path(entry.digest).is_file()

    def test_a_refused_artifact_records_no_result(self, tmp_path: Path) -> None:
        """No half-stored facts: if the outputs cannot be kept, the result
        is not recorded either."""
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        (run_dir / "metrics.json").unlink()
        store = FileEvidenceStore(tmp_path / "store")

        with pytest.raises(Exception, match="not a file"):
            store.record_result(result)

        assert store.results() == ()
        assert list((tmp_path / "store" / "results").glob("*.json")) == []


class TestPayloadDigest:
    def test_an_edited_result_fails_even_though_its_id_still_derives(
        self, tmp_path: Path
    ) -> None:
        """A result's id comes from its job id alone, so recomputing it
        cannot detect an edited metric. The payload digest can."""
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        FileEvidenceStore(tmp_path / "store").record_result(result)
        path = tmp_path / "store" / "results" / f"{result.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["x"] = 999.0
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(EvidenceIntegrityError, match="was edited"):
            FileEvidenceStore(tmp_path / "store").get_result(result.id)

    def test_an_edited_evidence_metric_fails_the_same_way(
        self, tmp_path: Path
    ) -> None:
        """Evidence identity covers its observation but not its metrics."""
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        store = FileEvidenceStore(tmp_path / "store")
        store.record_result(result)
        evidence = store.record_evidence(make_evidence(result))
        path = tmp_path / "store" / "evidence" / f"{evidence.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics"]["x"] = 42.0
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(EvidenceIntegrityError, match="was edited"):
            FileEvidenceStore(tmp_path / "store").get_evidence(evidence.id)

    def test_a_malformed_record_fails_loudly(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        result = make_result(run_dir)
        FileEvidenceStore(tmp_path / "store").record_result(result)
        path = tmp_path / "store" / "results" / f"{result.id}.json"
        path.write_text("{ not json", encoding="utf-8")

        with pytest.raises(EvidenceIntegrityError, match="not valid JSON"):
            FileEvidenceStore(tmp_path / "store").get_result(result.id)
