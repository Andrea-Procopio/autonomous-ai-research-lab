from __future__ import annotations

import sys
from pathlib import Path

from autonomous_research_lab.core.experiment import ExperimentStatus
from autonomous_research_lab.execution.executor import ExperimentJob, JobStatus
from autonomous_research_lab.execution.local import LocalExecutor


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


def test_replications_of_one_spec_get_distinct_records(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path / "runs")
    first = script_job(tmp_path, WRITES_METRICS, config={"scale": 1})
    second = script_job(tmp_path, WRITES_METRICS, config={"scale": 1}, attempt=1)

    assert first.id != second.id
    assert executor.collect(executor.submit(first)).id != executor.collect(
        executor.submit(second)
    ).id
