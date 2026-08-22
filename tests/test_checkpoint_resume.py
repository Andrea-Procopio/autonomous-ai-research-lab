"""Checkpoint-resume (Task 7A.1): finish the seed the crash interrupted.

Three layers, bottom up: the stub template's checkpoint discipline (a
resumed run ends byte-identical to the uninterrupted one, and bytes
that do not hash to the pinned digest are refused, not retrained); the
dispatch policy (a killed seed is re-picked once, with the blob store's
verified copy handed over; everything else keeps the old fresh-seed
rule); and the whole seam through the real engineer and executor — one
job killed by the stub's deterministic crash knob, the next dispatch
resuming it to the exact metrics a clean run produces.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autonomous_research_lab.core.experiment import (
    Environment,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
)
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.roles.engineer import (
    ResumeSource,
    SeedPlan,
    default_plan,
)
from examples.vision_lab.catalog import fill_slot
from examples.vision_lab.checkpoints import CheckpointResume
from examples.vision_lab.composition import STUB_SLOT

TEMPLATE_PATH = (
    Path(__file__).parent.parent
    / "examples"
    / "vision_lab"
    / "templates"
    / "stub_trainer_v1.py"
)


def stub_source() -> str:
    source = TEMPLATE_PATH.read_text(encoding="utf-8")
    return fill_slot(source, STUB_SLOT).replace(
        "__ARL_PRIMARY_METRIC__", "delta"
    )


def run_template(
    script: Path, run_dir: Path, *, seed: int, config: dict[str, object]
) -> subprocess.CompletedProcess[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config))
    return subprocess.run(
        [sys.executable, str(script)],
        env={
            **os.environ,
            "ARL_RUN_DIR": str(run_dir),
            "ARL_CONFIG": str(config_path),
            "ARL_SEED": str(seed),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def script(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("template") / "experiment.py"
    path.write_text(stub_source(), encoding="utf-8")
    return path


# -- 1. the stub template's checkpoint discipline ------------------------------


class TestStubTemplate:
    def test_a_full_run_leaves_a_complete_checkpoint(
        self, script: Path, tmp_path: Path
    ) -> None:
        done = run_template(script, tmp_path / "a", seed=11, config={})
        assert done.returncode == 0, done.stderr
        checkpoint = json.loads(
            (tmp_path / "a" / "checkpoint.json").read_text()
        )
        assert checkpoint["steps_completed"] == 5
        assert checkpoint["seed"] == 11
        assert (tmp_path / "a" / "metrics.json").is_file()

    def test_the_crash_knob_leaves_a_partial_checkpoint_and_no_metrics(
        self, script: Path, tmp_path: Path
    ) -> None:
        died = run_template(
            script, tmp_path / "b", seed=11, config={"fail_after_step": 2}
        )
        assert died.returncode == 3
        checkpoint = json.loads(
            (tmp_path / "b" / "checkpoint.json").read_text()
        )
        assert checkpoint["steps_completed"] == 2
        assert not (tmp_path / "b" / "metrics.json").exists()

    def test_a_resumed_run_ends_exactly_where_a_clean_one_does(
        self, script: Path, tmp_path: Path
    ) -> None:
        clean = run_template(script, tmp_path / "clean", seed=11, config={})
        assert clean.returncode == 0, clean.stderr
        died = run_template(
            script, tmp_path / "dead", seed=11, config={"fail_after_step": 2}
        )
        assert died.returncode == 3
        checkpoint = tmp_path / "dead" / "checkpoint.json"
        resumed = run_template(
            script,
            tmp_path / "resumed",
            seed=11,
            config={
                "resume_checkpoint": str(checkpoint),
                "resume_checkpoint_sha256": sha256_of(checkpoint),
            },
        )
        assert resumed.returncode == 0, resumed.stderr
        assert (tmp_path / "resumed" / "metrics.json").read_text() == (
            tmp_path / "clean" / "metrics.json"
        ).read_text()

    def test_bytes_that_do_not_hash_to_the_pinned_digest_are_refused(
        self, script: Path, tmp_path: Path
    ) -> None:
        died = run_template(
            script, tmp_path / "dead", seed=11, config={"fail_after_step": 2}
        )
        assert died.returncode == 3
        checkpoint = tmp_path / "dead" / "checkpoint.json"
        refused = run_template(
            script,
            tmp_path / "resumed",
            seed=11,
            config={
                "resume_checkpoint": str(checkpoint),
                "resume_checkpoint_sha256": "0" * 64,
            },
        )
        assert refused.returncode != 0
        assert "does not hash" in refused.stderr
        assert not (tmp_path / "resumed" / "metrics.json").exists()

    def test_another_seeds_checkpoint_is_refused(
        self, script: Path, tmp_path: Path
    ) -> None:
        died = run_template(
            script, tmp_path / "dead", seed=11, config={"fail_after_step": 2}
        )
        assert died.returncode == 3
        checkpoint = tmp_path / "dead" / "checkpoint.json"
        refused = run_template(
            script,
            tmp_path / "resumed",
            seed=23,
            config={
                "resume_checkpoint": str(checkpoint),
                "resume_checkpoint_sha256": sha256_of(checkpoint),
            },
        )
        assert refused.returncode != 0
        assert "another seed" in refused.stderr


# -- 2. the dispatch policy ----------------------------------------------------


def _spec(seeds: tuple[int, ...] = (11, 23, 47)) -> ExperimentSpec:
    return ExperimentSpec(
        prediction_id="pred_0123456789abcdef",
        objective="contrast",
        procedure="train and probe",
        metrics=("delta",),
        seeds=seeds,
    )


def _recorded_failure(
    store: FileEvidenceStore,
    runs: Path,
    spec: ExperimentSpec,
    *,
    seed: int,
    job: str,
    checkpoint: bool = True,
    was_resume: bool = False,
) -> ExperimentResult:
    """One failed attempt, committed the way recovery commits it: run
    directory, manifest, and the result ingested into the store."""
    run_dir = runs / job
    run_dir.mkdir(parents=True)
    (run_dir / "stdout.log").write_text("")
    (run_dir / "stderr.log").write_text("")
    manifest: dict[str, str] = {}
    artifacts: tuple[str, ...] = ()
    if checkpoint:
        payload = json.dumps(
            {"encoder": [[0.0]], "loss": 1.0, "seed": seed, "steps_completed": 2}
        )
        (run_dir / "checkpoint.json").write_text(payload)
        manifest["checkpoint.json"] = hashlib.sha256(
            payload.encode()
        ).hexdigest()
        artifacts = (str(run_dir / "checkpoint.json"),)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    config: dict[str, str] = {"spec_id": spec.id}
    if was_resume:
        config["resume_checkpoint"] = "/somewhere/earlier"
        config["resume_checkpoint_sha256"] = "f" * 64
    result = ExperimentResult(
        spec_id=spec.id,
        job_id=job,
        status=ExperimentStatus.FAILED,
        command=("python", "experiment.py"),
        environment=Environment(python_version="3.11", platform="test"),
        metrics={},
        config=config,
        seed=seed,
        artifacts=artifacts,
        logs=(str(run_dir / "stdout.log"), str(run_dir / "stderr.log")),
        failure_reason="orphaned: the submitting process died",
    )
    return store.record_result(result)


def _success(spec: ExperimentSpec, seed: int) -> ExperimentResult:
    return ExperimentResult(
        spec_id=spec.id,
        job_id=f"job_ok_{seed}",
        status=ExperimentStatus.COMPLETED,
        command=("python", "experiment.py"),
        environment=Environment(python_version="3.11", platform="test"),
        metrics={"delta": 0.1},
        seed=seed,
    )


class TestDefaultPlan:
    def test_fresh_used_and_exhausted_seeds(self) -> None:
        spec = _spec()
        assert default_plan(spec, ()).seed == 11
        assert default_plan(spec, (_success(spec, 11),)).seed == 23
        exhausted = tuple(_success(spec, seed) for seed in spec.seeds)
        assert default_plan(spec, exhausted).seed == 11
        assert default_plan(_spec(seeds=()), ()).seed is None

    def test_a_resume_names_its_seed(self) -> None:
        with pytest.raises(ValueError, match="names the seed"):
            SeedPlan(
                seed=None,
                resume=ResumeSource(
                    checkpoint="/x", sha256="0" * 64, from_job="job_x"
                ),
            )


class TestCheckpointResumePolicy:
    def test_a_killed_seed_is_resumed_from_the_blob_store(
        self, tmp_path: Path
    ) -> None:
        store = FileEvidenceStore(tmp_path / "evidence")
        spec = _spec()
        failed = _recorded_failure(
            store, tmp_path / "runs", spec, seed=11, job="job_dead"
        )
        plan = CheckpointResume(evidence=store).plan(spec, (failed,))
        assert plan.seed == 11
        assert plan.resume is not None
        assert plan.resume.from_job == "job_dead"
        blob = Path(plan.resume.checkpoint)
        assert blob.is_file()
        assert (
            hashlib.sha256(blob.read_bytes()).hexdigest()
            == plan.resume.sha256
        )
        # The handed-over copy is the blob store's, not the run dir's.
        assert str(tmp_path / "runs") not in plan.resume.checkpoint

    def test_a_succeeded_seed_is_never_resumed(self, tmp_path: Path) -> None:
        store = FileEvidenceStore(tmp_path / "evidence")
        spec = _spec()
        failed = _recorded_failure(
            store, tmp_path / "runs", spec, seed=11, job="job_dead"
        )
        plan = CheckpointResume(evidence=store).plan(
            spec, (failed, _success(spec, 11))
        )
        assert plan.seed == 23 and plan.resume is None

    def test_one_resume_per_seed_ever(self, tmp_path: Path) -> None:
        store = FileEvidenceStore(tmp_path / "evidence")
        spec = _spec()
        first = _recorded_failure(
            store, tmp_path / "runs", spec, seed=11, job="job_dead"
        )
        second = _recorded_failure(
            store,
            tmp_path / "runs",
            spec,
            seed=11,
            job="job_dead_again",
            was_resume=True,
        )
        plan = CheckpointResume(evidence=store).plan(spec, (first, second))
        assert plan.seed == 23 and plan.resume is None

    def test_a_failure_without_a_checkpoint_consumes_the_seed(
        self, tmp_path: Path
    ) -> None:
        store = FileEvidenceStore(tmp_path / "evidence")
        spec = _spec()
        failed = _recorded_failure(
            store,
            tmp_path / "runs",
            spec,
            seed=11,
            job="job_dead",
            checkpoint=False,
        )
        plan = CheckpointResume(evidence=store).plan(spec, (failed,))
        assert plan.seed == 23 and plan.resume is None

    def test_a_missing_blob_consumes_the_seed(self, tmp_path: Path) -> None:
        store = FileEvidenceStore(tmp_path / "evidence")
        spec = _spec()
        failed = _recorded_failure(
            store, tmp_path / "runs", spec, seed=11, job="job_dead"
        )
        resumable = CheckpointResume(evidence=store).plan(spec, (failed,))
        assert resumable.resume is not None
        Path(resumable.resume.checkpoint).unlink()
        plan = CheckpointResume(evidence=store).plan(spec, (failed,))
        assert plan.seed == 23 and plan.resume is None

    def test_a_fresh_spec_gets_the_default_plan(self, tmp_path: Path) -> None:
        store = FileEvidenceStore(tmp_path / "evidence")
        spec = _spec()
        plan = CheckpointResume(evidence=store).plan(spec, ())
        assert plan.seed == 11 and plan.resume is None


# -- 3. the whole seam: killed job, resumed by the next dispatch ---------------


class TestEngineerResumesTheKilledSeed:
    def test_the_next_dispatch_finishes_what_the_crash_interrupted(
        self, tmp_path: Path
    ) -> None:
        from autonomous_research_lab.core.actions import (
            ResearchAction,
            ResearchActionType,
        )
        from autonomous_research_lab.core.proposals import (
            ProposalKind,
            ResultProposal,
        )
        from autonomous_research_lab.execution.binding import (
            HostPythonBinding,
        )
        from autonomous_research_lab.execution.executor import ExperimentJob
        from autonomous_research_lab.execution.local import LocalExecutor
        from autonomous_research_lab.execution.runner import DirectJobRunner
        from autonomous_research_lab.roles.base import (
            RoleContext,
            RoleInvocation,
            RoleName,
        )
        from autonomous_research_lab.roles.engineer import (
            ImplementationTemplate,
            ModelBackedEngineer,
        )
        from autonomous_research_lab.runtime.implementation_store import (
            ImplementationStore,
        )
        from autonomous_research_lab.runtime.providers import (
            FakeModelProvider,
            UsageLedger,
        )

        source = stub_source()
        spec = _spec()
        store = FileEvidenceStore(tmp_path / "evidence")
        executor = LocalExecutor(tmp_path / "runs")

        # A clean full run of the same seed, for the ground truth.
        script = tmp_path / "clean.py"
        script.write_text(source, encoding="utf-8")
        clean = run_template(script, tmp_path / "clean", seed=11, config={})
        assert clean.returncode == 0, clean.stderr
        truth = json.loads((tmp_path / "clean" / "metrics.json").read_text())

        # The killed attempt: the stub's deterministic crash knob stands
        # in for kill -9 mid-training; the executor reaps it as FAILED
        # with the checkpoint collected, and the store ingests it.
        dead = executor.submit(
            ExperimentJob(
                spec_id=spec.id,
                command=(sys.executable, str(script)),
                config={"spec_id": spec.id, "fail_after_step": 2},
                seed=11,
                timeout_seconds=60.0,
            )
        )
        failed = executor.collect(dead)
        assert failed.status is ExperimentStatus.FAILED
        failed = store.record_result(failed)

        # The next dispatch: the engineer's policy re-picks seed 11 and
        # hands the new job the verified checkpoint.
        engineer = ModelBackedEngineer(
            provider=FakeModelProvider(
                (
                    json.dumps(
                        {
                            "files": [
                                {"path": "experiment.py", "content": source}
                            ],
                            "rationale": "resume fixture",
                        }
                    ),
                )
            ),
            model="test-model",
            runner=DirectJobRunner(executor),
            ledger=UsageLedger(),
            store=ImplementationStore(tmp_path / "implementations"),
            binding=HostPythonBinding(timeout_seconds=60.0),
            template=ImplementationTemplate(name="stub", source=source),
            dispatch=CheckpointResume(evidence=store),
        )
        action = ResearchAction(
            action_type=ResearchActionType.RUN_EXPERIMENT,
            rationale="assigned",
            targets=(spec.id,),
        )
        proposals = engineer.perform(
            RoleInvocation(
                role=RoleName.RESEARCH_ENGINEER,
                assignment=action,
                context=RoleContext(
                    objective="finish the family",
                    experiments=(spec,),
                    results=(failed,),
                ),
                allowed_actions=frozenset(
                    {ResearchActionType.RUN_EXPERIMENT}
                ),
                expected_output=frozenset({ProposalKind.RESULT}),
            )
        )
        assert len(proposals) == 1
        proposal = proposals[0]
        assert isinstance(proposal, ResultProposal)
        result = proposal.result
        assert result.seed == 11
        assert result.succeeded
        assert result.config["resumed_from_job"] == failed.job_id
        assert str(result.config["resume_checkpoint"]).startswith(
            str(tmp_path / "evidence")
        )
        assert dict(result.metrics) == truth
