from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import ExperimentStatus
from autonomous_research_lab.execution.executor import (
    DuplicateJobError,
    ExperimentJob,
    JobNotFinishedError,
    JobStatus,
    UnknownJobError,
    derive_job_id,
)
from autonomous_research_lab.execution.local import (
    JOB_RECORD_FILENAME,
    LocalExecutor,
    MalformedJobRecordError,
)


def script_job(tmp_path: Path, body: str, **overrides: object) -> ExperimentJob:
    script = tmp_path / "script.py"
    script.write_text(body)
    defaults: dict[str, object] = {
        "spec_id": "exp_1",
        "command": (sys.executable, str(script)),
        "working_dir": str(tmp_path),
        "seed": 3,
    }
    return ExperimentJob(**(defaults | overrides))  # type: ignore[arg-type]


WRITES_METRICS = """
import json, os
from pathlib import Path
run_dir = Path(os.environ["ARL_RUN_DIR"])
config = json.loads(Path(os.environ["ARL_CONFIG"]).read_text())
(run_dir / "metrics.json").write_text(json.dumps({
    "score": 0.25, "seed_seen": int(os.environ["ARL_SEED"]), "scale": config["scale"],
}))
"""


def test_metrics_come_from_the_file_the_process_wrote(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path / "runs")
    job = script_job(tmp_path, WRITES_METRICS, config={"scale": 4})

    job_id = executor.submit(job)
    assert executor.status(job_id) is JobStatus.SUCCEEDED

    result = executor.collect(job_id)
    assert result.status is ExperimentStatus.COMPLETED
    assert result.metrics == {"score": 0.25, "seed_seen": 3.0, "scale": 4.0}
    assert result.config["scale"] == 4
    assert any(a.endswith("metrics.json") for a in result.artifacts)


def test_a_crashing_experiment_produces_a_recorded_failure(tmp_path: Path) -> None:
    """A failed run is an outcome with provenance, not a gap. Returning a
    result here is what keeps failures analysable instead of invisible."""
    executor = LocalExecutor(tmp_path / "runs")
    job_id = executor.submit(script_job(tmp_path, "raise SystemExit(3)"))

    assert executor.status(job_id) is JobStatus.FAILED
    result = executor.collect(job_id)
    assert result.status is ExperimentStatus.FAILED
    assert result.exit_code == 3
    assert result.failure_reason is not None
    assert result.metrics == {}
    assert result.logs


def test_a_silent_experiment_is_a_failure(tmp_path: Path) -> None:
    """Exiting zero without writing metrics is not success. Treating it as
    success is how empty runs turn into reported findings."""
    executor = LocalExecutor(tmp_path / "runs")
    job_id = executor.submit(script_job(tmp_path, "print('nothing measured')"))

    assert executor.status(job_id) is JobStatus.FAILED
    assert "metrics.json" in str(executor.collect(job_id).failure_reason)


def test_non_numeric_metrics_are_rejected(tmp_path: Path) -> None:
    body = (
        "import json, os\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARL_RUN_DIR'], 'metrics.json')"
        ".write_text(json.dumps({'score': 'high'}))\n"
    )
    executor = LocalExecutor(tmp_path / "runs")
    job_id = executor.submit(script_job(tmp_path, body))

    assert executor.status(job_id) is JobStatus.FAILED
    assert "not a number" in str(executor.collect(job_id).failure_reason)


class TestOccurrenceSemantics:
    def test_identical_executions_yield_distinct_records(self, tmp_path: Path) -> None:
        """Two identically configured runs of one spec are two events with two
        records — replications never collide."""
        executor = LocalExecutor(tmp_path / "runs")
        first = script_job(tmp_path, WRITES_METRICS, config={"scale": 1})
        second = script_job(tmp_path, WRITES_METRICS, config={"scale": 1})

        assert first.id != second.id
        first_result = executor.collect(executor.submit(first))
        second_result = executor.collect(executor.submit(second))
        assert first_result.id != second_result.id
        assert first_result.metrics == second_result.metrics

    def test_a_job_submits_at_most_once(self, tmp_path: Path) -> None:
        """Resubmitting the same job object would silently overwrite its run
        directory and record; a retry must be a new job."""
        executor = LocalExecutor(tmp_path / "runs")
        job = script_job(tmp_path, WRITES_METRICS, config={"scale": 1})
        executor.submit(job)
        with pytest.raises(DuplicateJobError):
            executor.submit(job)


# -- the durable job record ----------------------------------------------------


class TestAColdProcessCanFindTheJob:
    """The executor's in-memory maps die with the process. The run
    directory does not, and after a crash it is all that is left."""

    def test_a_fresh_executor_collects_a_job_it_did_not_submit(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        job = script_job(tmp_path, WRITES_METRICS, config={"scale": 4})
        job_id = LocalExecutor(runs).submit(job)

        cold = LocalExecutor(runs)  # nothing in memory, as after a restart

        assert cold.status(job_id) is JobStatus.SUCCEEDED
        result = cold.collect(job_id)
        assert result.status is ExperimentStatus.COMPLETED
        assert result.metrics["score"] == 0.25
        assert result.seed == 3
        assert result.job_id == job_id

    def test_the_reconstructed_result_is_the_one_that_was_collected(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        warm = LocalExecutor(runs)
        job_id = warm.submit(
            script_job(tmp_path, WRITES_METRICS, config={"scale": 4})
        )

        assert LocalExecutor(runs).collect(job_id) == warm.collect(job_id)

    def test_a_failed_job_is_reconstructed_as_a_failure(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        job_id = LocalExecutor(runs).submit(
            script_job(tmp_path, "raise SystemExit(3)")
        )

        result = LocalExecutor(runs).collect(job_id)

        assert result.status is ExperimentStatus.FAILED
        assert result.exit_code == 3
        assert result.failure_reason is not None

    def test_a_fresh_executor_refuses_to_run_the_job_again(
        self, tmp_path: Path
    ) -> None:
        """The point of the whole mechanism: a recovered run collects, it
        never resubmits."""
        runs = tmp_path / "runs"
        job = script_job(tmp_path, WRITES_METRICS, config={"scale": 4})
        LocalExecutor(runs).submit(job)

        with pytest.raises(DuplicateJobError):
            LocalExecutor(runs).submit(job)

    def test_an_unknown_job_is_still_unknown(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownJobError):
            LocalExecutor(tmp_path / "runs").status("job_nowhere")

    def test_a_job_whose_process_died_is_running_and_uncollectable(
        self, tmp_path: Path
    ) -> None:
        """What a killed run leaves behind: a record saying the job began
        and nothing saying how it ended."""
        runs = tmp_path / "runs"
        job = script_job(tmp_path, WRITES_METRICS, config={"scale": 4})
        LocalExecutor(runs)._write_record(job, JobStatus.RUNNING, None)

        cold = LocalExecutor(runs)

        assert cold.status(job.id) is JobStatus.RUNNING
        with pytest.raises(JobNotFinishedError, match="left no result"):
            cold.collect(job.id)

    def test_an_unreadable_record_is_loud_rather_than_absent(
        self, tmp_path: Path
    ) -> None:
        """Reporting it as unknown would invite resubmitting a job that
        may have run."""
        runs = tmp_path / "runs"
        job_id = LocalExecutor(runs).submit(
            script_job(tmp_path, WRITES_METRICS, config={"scale": 4})
        )
        (runs / job_id / JOB_RECORD_FILENAME).write_text("{not json")

        with pytest.raises(MalformedJobRecordError):
            LocalExecutor(runs).collect(job_id)

    def test_the_record_holds_no_environment_values(
        self, tmp_path: Path
    ) -> None:
        """``job.env`` is the one field a caller may fill with a secret,
        and a file on disk is where a secret must not be."""
        runs = tmp_path / "runs"
        job = script_job(
            tmp_path,
            WRITES_METRICS,
            config={"scale": 4},
            env={"SECRET_TOKEN": "hunter2"},
        )
        job_id = LocalExecutor(runs).submit(job)

        record = (runs / job_id / JOB_RECORD_FILENAME).read_text()

        assert "hunter2" not in record
        assert "SECRET_TOKEN" not in record

    def test_the_record_is_not_collected_as_an_artifact(
        self, tmp_path: Path
    ) -> None:
        """It is rewritten when the run ends, so hashing it into the
        manifest would guarantee a manifest that no longer matches."""
        runs = tmp_path / "runs"
        job_id = LocalExecutor(runs).submit(
            script_job(tmp_path, WRITES_METRICS, config={"scale": 4})
        )

        result = LocalExecutor(runs).collect(job_id)

        assert not any(
            Path(artifact).name == JOB_RECORD_FILENAME
            for artifact in result.artifacts
        )


class TestDerivedJobIds:
    def test_a_job_id_is_the_same_in_every_process(self) -> None:
        assert derive_job_id("att_1") == derive_job_id("att_1")

    def test_two_attempts_never_share_a_job(self) -> None:
        assert derive_job_id("att_1") != derive_job_id("att_2")

    def test_an_attempt_is_required(self) -> None:
        with pytest.raises(ValueError, match="attempt"):
            derive_job_id("  ")

    def test_a_supplied_id_is_the_job_id(self, tmp_path: Path) -> None:
        job = script_job(
            tmp_path, WRITES_METRICS, config={"scale": 4}, id=derive_job_id("att_1")
        )

        assert job.id == derive_job_id("att_1")
