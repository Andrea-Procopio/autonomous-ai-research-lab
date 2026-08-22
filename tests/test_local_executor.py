from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import Environment, ExperimentStatus
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
    _JobFacts,
)


def running_facts(
    job: ExperimentJob, *, started_at: float = 0.0, pid: int | None = None
) -> _JobFacts:
    """The record a submitter writes before launching ``job`` — what a
    killed run leaves behind for a cold process to find."""
    return _JobFacts(
        job_id=job.id,
        spec_id=job.spec_id,
        command=job.command,
        config=dict(job.config),
        seed=job.seed,
        required_artifacts=job.required_artifacts,
        timeout_seconds=job.timeout_seconds,
        gpu_count=job.gpu_count,
        environment=Environment(python_version="3", platform="test"),
        started_at=started_at,
        pid=pid,
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
        LocalExecutor(runs)._write_record(
            running_facts(job), JobStatus.RUNNING, None
        )

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


def test_gpu_occupancy_is_billed_from_the_job(tmp_path: Path) -> None:
    """gpu_hours = wall clock x declared occupancy — what the lab could
    not schedule elsewhere, whatever the kernels achieved."""
    runs = tmp_path / "runs"
    job = script_job(
        tmp_path, WRITES_METRICS, config={"scale": 1}, gpu_count=2
    )
    executor = LocalExecutor(runs)

    result = executor.collect(executor.submit(job))

    assert result.cost.gpu_hours == pytest.approx(
        result.cost.wall_clock_seconds * 2 / 3600.0
    )


def test_a_cpu_job_bills_no_gpu_hours(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    executor = LocalExecutor(runs)
    result = executor.collect(
        executor.submit(script_job(tmp_path, WRITES_METRICS, config={"scale": 1}))
    )
    assert result.cost.gpu_hours == 0.0


class TestReapingOrphans:
    """A job whose submitter died is finalized from its own evidence."""

    def dead_pid(self) -> int:
        """A pid that provably belonged to a process that is gone."""
        import subprocess

        process = subprocess.Popen((sys.executable, "-c", "pass"))
        process.wait()
        return process.pid

    def plant_orphan(
        self,
        tmp_path: Path,
        *,
        with_metrics: bool = True,
        pid: int | None = None,
        gpu_count: int = 0,
    ) -> tuple[LocalExecutor, str]:
        """A run directory and a RUNNING record, as a killed submitter
        leaves them: the job launched, the record never rewritten."""
        runs = tmp_path / "runs"
        job = script_job(
            tmp_path,
            WRITES_METRICS,
            config={"scale": 1},
            gpu_count=gpu_count,
        )
        run_dir = runs / job.id
        run_dir.mkdir(parents=True)
        if with_metrics:
            (run_dir / "metrics.json").write_text('{"value": 4.0}')
        executor = LocalExecutor(runs)
        executor._write_record(
            running_facts(
                job,
                started_at=run_dir.stat().st_mtime - 30.0,
                pid=self.dead_pid() if pid is None else pid,
            ),
            JobStatus.RUNNING,
            None,
        )
        return LocalExecutor(runs), job.id

    def test_a_finished_orphan_is_reaped_as_a_success(
        self, tmp_path: Path
    ) -> None:
        executor, job_id = self.plant_orphan(tmp_path)

        assert executor.reap(job_id) is JobStatus.SUCCEEDED

        result = LocalExecutor(tmp_path / "runs").collect(job_id)
        assert result.status is ExperimentStatus.COMPLETED
        assert result.metrics == {"value": 4.0}
        assert result.exit_code is None  # nobody watched it end
        assert result.cost.wall_clock_seconds > 0.0

    def test_an_orphan_without_metrics_is_reaped_as_a_failure(
        self, tmp_path: Path
    ) -> None:
        executor, job_id = self.plant_orphan(tmp_path, with_metrics=False)

        assert executor.reap(job_id) is JobStatus.FAILED

        result = LocalExecutor(tmp_path / "runs").collect(job_id)
        assert result.failure_reason is not None
        assert result.failure_reason.startswith(
            "orphaned: the submitting process died"
        )

    def test_a_live_pid_refuses_the_reap(self, tmp_path: Path) -> None:
        """Alive — or reused: probing cannot tell, and both refuse."""
        import os

        executor, job_id = self.plant_orphan(tmp_path, pid=os.getpid())

        assert executor.reap(job_id) is JobStatus.RUNNING
        with pytest.raises(JobNotFinishedError):
            LocalExecutor(tmp_path / "runs").collect(job_id)

    def test_a_record_without_a_pid_refuses_the_reap(
        self, tmp_path: Path
    ) -> None:
        """The crash landed between the two record writes; nothing can be
        proven about the process, so nothing is finalized."""
        runs = tmp_path / "runs"
        job = script_job(tmp_path, WRITES_METRICS, config={"scale": 1})
        (runs / job.id).mkdir(parents=True)
        executor = LocalExecutor(runs)
        executor._write_record(running_facts(job), JobStatus.RUNNING, None)

        assert executor.reap(job.id) is JobStatus.RUNNING

    def test_an_old_format_record_refuses_the_reap(
        self, tmp_path: Path
    ) -> None:
        """Everything in live_runs/ predates these fields; a record that
        cannot prove a death is left exactly as found."""
        import json

        runs = tmp_path / "runs"
        job = script_job(tmp_path, WRITES_METRICS, config={"scale": 1})
        run_dir = runs / job.id
        run_dir.mkdir(parents=True)
        (run_dir / JOB_RECORD_FILENAME).write_text(
            json.dumps(
                {
                    "job_id": job.id,
                    "spec_id": job.spec_id,
                    "status": "running",
                    "command": list(job.command),
                    "config": dict(job.config),
                    "seed": job.seed,
                }
            )
        )

        assert LocalExecutor(runs).reap(job.id) is JobStatus.RUNNING

    def test_reaping_a_terminal_job_is_a_no_op(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        executor = LocalExecutor(runs)
        job_id = executor.submit(
            script_job(tmp_path, WRITES_METRICS, config={"scale": 1})
        )
        before = (runs / job_id / JOB_RECORD_FILENAME).read_text()

        assert LocalExecutor(runs).reap(job_id) is JobStatus.SUCCEEDED
        assert (runs / job_id / JOB_RECORD_FILENAME).read_text() == before

    def test_reaping_twice_is_reaping_once(self, tmp_path: Path) -> None:
        executor, job_id = self.plant_orphan(tmp_path)
        executor.reap(job_id)
        record = (tmp_path / "runs" / job_id / JOB_RECORD_FILENAME).read_text()

        assert executor.reap(job_id) is JobStatus.SUCCEEDED
        assert (
            tmp_path / "runs" / job_id / JOB_RECORD_FILENAME
        ).read_text() == record

    def test_a_reaped_gpu_job_bills_its_occupancy(
        self, tmp_path: Path
    ) -> None:
        executor, job_id = self.plant_orphan(tmp_path, gpu_count=2)
        executor.reap(job_id)

        result = LocalExecutor(tmp_path / "runs").collect(job_id)
        assert result.cost.gpu_hours == pytest.approx(
            result.cost.wall_clock_seconds * 2 / 3600.0
        )

    def test_the_charge_is_clamped_to_the_authorization(
        self, tmp_path: Path
    ) -> None:
        """The artifacts' clock can claim anything; the executor never
        charges beyond the timeout it authorized."""
        runs = tmp_path / "runs"
        job = script_job(
            tmp_path,
            WRITES_METRICS,
            config={"scale": 1},
            timeout_seconds=10.0,
        )
        run_dir = runs / job.id
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text('{"value": 1.0}')
        executor = LocalExecutor(runs)
        executor._write_record(
            running_facts(
                job,
                started_at=run_dir.stat().st_mtime - 9_999.0,
                pid=self.dead_pid(),
            ),
            JobStatus.RUNNING,
            None,
        )

        executor.reap(job.id)

        result = LocalExecutor(runs).collect(job.id)
        assert result.cost.wall_clock_seconds == pytest.approx(10.0)


ORPHAN_SUBMITTER = """
import sys
from pathlib import Path

sys.path.insert(0, {src!r})
from autonomous_research_lab.execution.executor import ExperimentJob
from autonomous_research_lab.execution.local import LocalExecutor

job = ExperimentJob(
    spec_id="exp_orphan",
    command=(sys.executable, {script!r}),
    working_dir={workdir!r},
    seed=3,
    id="job_orphan",
)
LocalExecutor({runs!r}).submit(job)
"""

SLEEPS_THEN_WRITES = """
import json, os, time
from pathlib import Path

time.sleep(1.5)
run_dir = Path(os.environ["ARL_RUN_DIR"])
(run_dir / "metrics.json").write_text(json.dumps({"value": 4.0}))
"""


def test_a_job_that_finished_after_its_submitter_died_is_reaped(
    tmp_path: Path,
) -> None:
    """The real story, with real processes: the submitter is SIGKILLed,
    the job — in its own session — survives and finishes, and a cold
    executor closes the books on it."""
    import signal
    import subprocess
    import time

    src = str(Path(__file__).resolve().parent.parent / "src")
    runs = tmp_path / "runs"
    script = tmp_path / "experiment.py"
    script.write_text(SLEEPS_THEN_WRITES)
    submitter_code = ORPHAN_SUBMITTER.format(
        src=src,
        script=str(script),
        workdir=str(tmp_path),
        runs=str(runs),
    )

    submitter = subprocess.Popen((sys.executable, "-c", submitter_code))
    record = runs / "job_orphan" / JOB_RECORD_FILENAME
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if record.is_file() and '"pid":' in record.read_text():
            import json

            if json.loads(record.read_text()).get("pid") is not None:
                break
        time.sleep(0.02)
    else:
        raise AssertionError("the submitter never recorded a pid")

    submitter.send_signal(signal.SIGKILL)
    submitter.wait()

    metrics = runs / "job_orphan" / "metrics.json"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not metrics.is_file():
        time.sleep(0.05)
    assert metrics.is_file(), "the orphaned job never finished"
    # The job process itself must be gone before the reap can prove death.
    import json

    pid = json.loads(record.read_text())["pid"]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)

    cold = LocalExecutor(runs)
    assert cold.reap("job_orphan") is JobStatus.SUCCEEDED
    result = cold.collect("job_orphan")
    assert result.metrics == {"value": 4.0}
    assert result.failure_reason is None
