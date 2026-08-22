"""Checkpoint-resume: finish the seed the crash interrupted.

Task 7A.1. Recovery already does the durable half: a job whose process
died half-trained is reaped, its checkpoint file is collected, hashed
into the run's manifest, and ingested into the evidence blob store when
the failed result commits. What was missing is the other half — the
next dispatch of the same spec re-picking the killed seed and handing
the new job that checkpoint, instead of consuming the seed and losing
the family member.

The policy is trusted code and reads only durable records. The
checkpoint handed over is the *blob store's* copy — content-addressed,
covered by ``arl verify`` — never the dead job's mutable run directory,
and the digest travels in the job config so the template refuses bytes
that do not hash to what the record pins.

Bounded by construction: a failed attempt that was itself a resume
never offers its checkpoint, so a seed is resumed at most once — after
that it is consumed exactly as before this policy existed. Host
backends only: the blob path is a host path; mounting it into a
container is deliberately not built yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from autonomous_research_lab.core.experiment import (
    ExperimentResult,
    ExperimentSpec,
)
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.roles.engineer import (
    ResumeSource,
    SeedPlan,
    default_plan,
)

#: The names a template may give its checkpoint file, at the run
#: directory's root. Mirrored in each template's fixed region — the
#: templates cannot import this package.
CHECKPOINT_FILENAMES: Final = ("checkpoint.pt", "checkpoint.json")

_RESUME_KEY: Final = "resume_checkpoint"


@dataclass(frozen=True, slots=True)
class CheckpointResume:
    """The dispatch policy: resume a killed seed from its verified
    checkpoint, else fall back to the default fresh-seed plan."""

    evidence: FileEvidenceStore

    def plan(
        self, spec: ExperimentSpec, results: tuple[ExperimentResult, ...]
    ) -> SeedPlan:
        mine = tuple(r for r in results if r.spec_id == spec.id)
        for seed in spec.seeds:
            seeded = [r for r in mine if r.seed == seed]
            if not seeded or any(r.succeeded for r in seeded):
                continue
            if any(_RESUME_KEY in r.config for r in seeded):
                # One resume per seed, ever: a resumed attempt that
                # failed too consumes the seed, exactly as before.
                continue
            source = self._checkpoint_of(seeded[-1])
            if source is not None:
                return SeedPlan(seed=seed, resume=source)
        return default_plan(spec, mine)

    def _checkpoint_of(self, result: ExperimentResult) -> ResumeSource | None:
        """The newest failed attempt's ingested checkpoint, as the blob
        store's verified copy. ``None`` when the run left no checkpoint
        or nothing was ingested — that seed stays consumed."""
        manifest = self.evidence.artifacts.get(result.id)
        if manifest is None:
            return None
        for entry in manifest.entries:
            if entry.path not in CHECKPOINT_FILENAMES:
                continue
            blob = self.evidence.artifacts.blob_path(entry.digest)
            if not blob.is_file():
                return None
            return ResumeSource(
                checkpoint=str(blob),
                sha256=entry.digest,
                from_job=result.job_id,
            )
        return None
