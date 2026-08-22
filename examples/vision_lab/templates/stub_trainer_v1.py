"""The stub trainer: the encoder contrast's metric shape, stdlib only.

The trusted starting point the engineer's model completes when the lab
runs in CI: no torch, no dataset, seconds of wall clock — and byte-for-
byte the same *contract* as the real trainer. A seeded synthetic
"image" distribution stands in for CIFAR; everything that measures is
fixed code; the one slot is the encoder the model builds.

Contract with the executor:
  reads  ARL_RUN_DIR, ARL_CONFIG, ARL_SEED
  writes $ARL_RUN_DIR/metrics.json

The string ``__ARL_PRIMARY_METRIC__`` is replaced by trusted catalog
code with the admitted prediction's verbatim metric key before the
model ever sees this source.
"""

# ARL-FIXED-BEGIN contract
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

PRIMARY_METRIC_KEY = "__ARL_PRIMARY_METRIC__"
CHECKPOINT_FILENAME = "checkpoint.json"

DIM = 16
N_CLASSES = 4
N_TRAIN = 160
N_PROBE_TRAIN = 200
N_PROBE_EVAL = 120
TINY_SUBSET = 32
TRAIN_STEPS = 5


def example(rng: random.Random, label: int) -> list[float]:
    """One synthetic 'image': a class-dependent direction plus noise."""
    return [
        math.sin(1.7 * label + 0.9 * axis) + rng.gauss(0.0, 1.6)
        for axis in range(DIM)
    ]


def batch(
    rng: random.Random, count: int
) -> tuple[list[list[float]], list[int]]:
    labels = [rng.randrange(N_CLASSES) for _ in range(count)]
    return [example(rng, label) for label in labels], labels


def encode(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(weight * value for weight, value in zip(row, vector, strict=False))
        for row in matrix
    ]


def train(
    matrix: list[list[float]],
    data: list[list[float]],
    labels: list[int],
    *,
    run_dir: Path,
    seed: int,
    start_step: int = 0,
    loss: float = 0.0,
    fail_after: int | None = None,
) -> tuple[list[list[float]], float]:
    """Deterministic 'training': pull each row toward the centroid of
    one class, a few steps. The point is a real, seeded process whose
    output beats an untrained one — not deep learning.

    Every completed step writes a checkpoint carrying the matrix, the
    step count, the running loss, and the seed — so a killed run can be
    resumed from ``start_step`` and end in exactly the state the
    uninterrupted run would have reached. ``fail_after`` is the stub's
    deterministic stand-in for a crash: exit 3 after that many steps,
    checkpoint already durable.
    """
    trained = [list(row) for row in matrix]
    for step in range(start_step, TRAIN_STEPS):
        loss = 0.0
        for target in range(min(N_CLASSES, len(trained))):
            members = [
                vector
                for vector, label in zip(data, labels, strict=False)
                if label == target
            ]
            if not members:
                continue
            centroid = [
                sum(values) / len(members) for values in zip(*members, strict=False)
            ]
            row = trained[target]
            loss += sum(
                (weight - value) ** 2
                for weight, value in zip(row, centroid, strict=False)
            )
            trained[target] = [
                weight + 0.5 * (value - weight)
                for weight, value in zip(row, centroid, strict=False)
            ]
        (run_dir / CHECKPOINT_FILENAME).write_text(
            json.dumps(
                {
                    "encoder": trained,
                    "loss": loss,
                    "seed": seed,
                    "steps_completed": step + 1,
                },
                sort_keys=True,
            )
        )
        if fail_after is not None and step + 1 >= fail_after:
            raise SystemExit(3)
    return trained, loss


def resume_state(
    config: dict[str, object], seed: int
) -> tuple[list[list[float]] | None, int, float]:
    """The checkpoint the config hands over, verified before trust:
    the bytes must hash to the digest the record pins, and the
    checkpoint must belong to this seed. A mismatch is a refusal, not
    a fresh start — training silently from scratch would misreport a
    resumed run as one."""
    named = config.get("resume_checkpoint")
    if not isinstance(named, str) or not named:
        return None, 0, 0.0
    declared = config.get("resume_checkpoint_sha256")
    raw = Path(named).read_bytes()
    if (
        not isinstance(declared, str)
        or hashlib.sha256(raw).hexdigest() != declared
    ):
        raise SystemExit(
            "resume checkpoint does not hash to the digest the record pins"
        )
    payload = json.loads(raw)
    if int(payload["seed"]) != seed:
        raise SystemExit("resume checkpoint belongs to another seed")
    matrix = [[float(value) for value in row] for row in payload["encoder"]]
    return matrix, int(payload["steps_completed"]), float(payload["loss"])


def probe_accuracy(
    matrix: list[list[float]],
    train_data: list[list[float]],
    train_labels: list[int],
    eval_data: list[list[float]],
    eval_labels: list[int],
) -> float:
    """Nearest-centroid probe in the encoded space: fit on one split,
    score on another, exactly like the linear probe it stands in for."""
    encoded = [encode(matrix, vector) for vector in train_data]
    centroids: dict[int, list[float]] = {}
    for target in range(N_CLASSES):
        members = [
            vector
            for vector, label in zip(encoded, train_labels, strict=False)
            if label == target
        ]
        if members:
            centroids[target] = [
                sum(values) / len(members) for values in zip(*members, strict=False)
            ]
    hits = 0
    for vector, label in zip(eval_data, eval_labels, strict=False):
        coded = encode(matrix, vector)
        nearest = min(
            centroids,
            key=lambda target: sum(
                (a - b) ** 2 for a, b in zip(coded, centroids[target], strict=False)
            ),
        )
        hits += int(nearest == label)
    return hits / len(eval_data)


def main() -> None:
    run_dir = Path(os.environ["ARL_RUN_DIR"])
    seed = int(os.environ.get("ARL_SEED", "0"))
    config_path = os.environ.get("ARL_CONFIG")
    config: dict[str, object] = {}
    if config_path and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text())

    rng = random.Random(seed)
    trained_encoder = build_encoder(random.Random(seed + 1))
    random_encoder = build_encoder(random.Random(seed + 2))

    resumed, start_step, prior_loss = resume_state(config, seed)
    if resumed is not None:
        trained_encoder = resumed
    fail_after = config.get("fail_after_step")

    train_data, train_labels = batch(rng, N_TRAIN)
    trained_encoder, final_loss = train(
        trained_encoder,
        train_data,
        train_labels,
        run_dir=run_dir,
        seed=seed,
        start_step=start_step,
        loss=prior_loss,
        fail_after=int(fail_after) if isinstance(fail_after, int) else None,
    )

    probe_train, probe_train_labels = batch(rng, N_PROBE_TRAIN)
    probe_eval, probe_eval_labels = batch(rng, N_PROBE_EVAL)
    trained_top1 = probe_accuracy(
        trained_encoder,
        probe_train,
        probe_train_labels,
        probe_eval,
        probe_eval_labels,
    )
    random_top1 = probe_accuracy(
        random_encoder,
        probe_train,
        probe_train_labels,
        probe_eval,
        probe_eval_labels,
    )

    # The positive control: the probe pipeline must fit a memorizable
    # subset scored on itself, or the instrument is broken. Noiseless
    # class prototypes, deliberately: separable by construction, so only
    # a broken encoder or a broken probe can miss them.
    tiny_labels = [index % N_CLASSES for index in range(TINY_SUBSET)]
    tiny = [
        [math.sin(1.7 * label + 0.9 * axis) for axis in range(DIM)]
        for label in tiny_labels
    ]
    overfit = probe_accuracy(
        trained_encoder, tiny, tiny_labels, tiny, tiny_labels
    )

    metrics = {
        PRIMARY_METRIC_KEY: trained_top1 - random_top1,
        "trained_encoder_probe_top1": trained_top1,
        "random_encoder_probe_top1": random_top1,
        "encoder_train_loss_final": final_loss,
        "tiny_subset_overfit_top1": overfit,
        "n_probe_train": N_PROBE_TRAIN,
        "n_probe_eval": N_PROBE_EVAL,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
# ARL-FIXED-END contract


# ARL-SLOT-BEGIN build_encoder
def build_encoder(rng: random.Random) -> list[list[float]]:
    """Return a DIM x DIM projection matrix, rows as lists, using only
    ``rng`` for randomness. Constraints: exactly DIM rows of DIM floats,
    every value finite, no imports beyond this file's, no I/O."""
    raise NotImplementedError("the engineer completes this slot")
# ARL-SLOT-END build_encoder


# ARL-FIXED-BEGIN entry
if __name__ == "__main__":
    main()
# ARL-FIXED-END entry
