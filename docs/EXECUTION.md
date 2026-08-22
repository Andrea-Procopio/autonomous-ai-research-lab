# Execution backends

Where experiment jobs run, and what an operator sets up before they can.

The invariant every backend keeps: **nothing scientific varies with the
choice.** The spec, the predictions, the state, and every gate decision
are byte-identical whether a job ran on a laptop CPU, an Apple GPU, or a
CUDA card. What differs is the job's command, its declared GPU
occupancy, its timeout, and the path at which its dataset appears — all
recorded as execution provenance in the job record, none of it reachable
from a scientific record.

## Profiles

A lab reads one deployment profile (JSON, operator-written, never stored
in any run record):

```json
{
  "backend": "container-cpu",
  "datasets_root": "/Users/you/arl-data",
  "image": "pytorch/pytorch@sha256:<digest>",
  "docker_host": "unix:///Users/you/.colima/default/docker.sock",
  "cpus": 4.0,
  "memory": "8g",
  "shm_size": "1g",
  "timeout_seconds": 900
}
```

| backend | where it runs | GPUs billed | notes |
| --- | --- | --- | --- |
| `host-cpu` | host interpreter | 0 | |
| `host-mps` | host interpreter | 1 | Apple silicon only |
| `host-cuda` | host interpreter | declared | Linux only |
| `container-cpu` | Docker/colima | 0 | the default for live runs |
| `container-cuda` | Docker on Linux | declared | no GPU passthrough under a macOS VM, ever |

Misconfigurations refuse at composition time — before any stage runs,
before any model call: a container backend without a digest-pinned
image, a CUDA backend on macOS, MPS anywhere but darwin.

**Generated code runs in a container.** That doctrine predates profiles
and survives them. A host backend composes with a model-completing
engineer only when the deployment file says
`"allow_generated_code_on_host": true` — an operator's recorded
decision, made for a machine they accept the consequences for, never a
default.

## Images

Container backends need a pre-pulled, digest-pinned image. The shim runs
with `--pull never` and `--network none`, so every dependency a job has
comes from the image and nothing else.

```bash
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime   # or any torch image
docker images --digests    # copy the sha256, write it into the profile
```

The same torch image serves `container-cpu` fine (CUDA libraries idle).
No digest is hardcoded in this repository: the operator pulled the
image, the operator names it, and the job record preserves what ran.

On macOS, colima supplies the daemon:

```bash
colima start
# profile: "docker_host": "unix:///Users/<you>/.colima/default/docker.sock"
```

## Datasets

Jobs run without network, so datasets are staged once, by the operator,
in trusted code, against a pinned archive digest:

```bash
python -m examples.vision_lab.stage_cifar10 --datasets-root ~/arl-data
```

Staging writes a write-once manifest — per-file sha256s, with an id
derived from them, so the same bytes staged on any machine get the same
id. A job declares `dataset_id` (machine-independent) and reads
`dataset_root` (the one value that legitimately differs by backend:
a host path, or `/arl/data/<name>` inside the container's read-only
mount). Preflight re-verifies the staged files against the manifest
before every launch; a dataset that no longer verifies fails the job
before anything runs or spends.

## GPU accounting

`gpu_hours` is billed as wall clock × declared occupancy — what the lab
could not schedule elsewhere while the job ran, not what the kernels
achieved. Occupancy is honest and defensible from the record;
utilization metering would be a measurement infrastructure of its own,
and the ledger never pretends to a number nobody took.

## Checkpoints and resume (Task 7A.1)

Templates write periodic checkpoints under the run directory — the
encoder template per epoch, the augmentation template per completed
arm, the stub per step — and they are collected and hashed like any
artifact. Policy at CIFAR scale: `state_dict` only, no optimizer state
— each file stays comfortably under the evidence store's 64 MiB blob
ceiling, and an oversized checkpoint fails ingest loudly, which is the
correct failure.

A job killed half-trained resumes at the next dispatch. Recovery
already commits the dead attempt as a failed result with its
checkpoint ingested; the engineer's dispatch policy
(`examples/vision_lab/checkpoints.py`) then re-picks the killed seed
and hands the new job the **blob store's** verified copy — never the
dead job's mutable run directory — with the sha256 pinned in the job
config. The template refuses bytes that do not hash to that digest,
and refuses another seed's checkpoint; the result's config records
`resumed_from_job`, so a resumed run never passes as an uninterrupted
one. Bounded: a failed attempt that was itself a resume never offers
its checkpoint, so a seed is resumed at most once and then consumed
exactly as before. Host backends only — mounting the blob into a
container is deliberately not built yet. No optimizer state travels,
so a resumed trajectory is its own honest measurement, not a replay of
the uninterrupted one; the stub trainer, which has no optimizer, ends
byte-identical either way and the tests pin that.
