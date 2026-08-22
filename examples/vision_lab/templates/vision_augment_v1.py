"""Augmented-versus-plain training contrast on CIFAR-10, under torch.

Two identical architectures, both trained supervised on the same seeded
subset; one sees flip-and-crop augmentation, the other the plain
images. Both are frozen and judged the same way: a linear probe fit on
their features. The admitted prediction is about the *difference* the
augmentation made, and everything that measures it — seeding, data,
splits, the probe, the control, the metrics file — is fixed code. The
one slot is the encoder architecture itself.

Contract with the executor:
  reads  ARL_RUN_DIR, ARL_CONFIG (dataset_root, optional smoke), ARL_SEED
  writes $ARL_RUN_DIR/metrics.json

The string ``__ARL_PRIMARY_METRIC__`` is replaced by trusted catalog
code with the admitted prediction's verbatim metric key before the
model ever sees this source. ``smoke: true`` in config shrinks every
count for an operator's minutes-long sanity pass.
"""

# ARL-FIXED-BEGIN contract
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset

PRIMARY_METRIC_KEY = "__ARL_PRIMARY_METRIC__"

N_ENCODER_TRAIN = 8_000
N_PROBE_TRAIN = 5_000
N_PROBE_EVAL = 2_000
TINY_SUBSET = 128
ENCODER_EPOCHS = 2
PROBE_EPOCHS = 300
BATCH = 128
N_CLASSES = 10


def device(config: dict[str, object]) -> torch.device:
    """The deployment's explicit choice, from config — never a guess.
    A template auto-detecting a GPU under a profile that declared none
    would bill a falsehood; absent a declaration, CPU is the only
    device nothing has to be declared for."""
    declared = config.get("device")
    if isinstance(declared, str) and declared:
        return torch.device(declared)
    return torch.device("cpu")


def features_of(
    encoder: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    where: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    banks, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            banks.append(encoder(images.to(where)).flatten(1).cpu())
            labels.append(targets)
    return torch.cat(banks), torch.cat(labels)


def probe_top1(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
) -> float:
    """A multinomial logistic probe, fit and scored deterministically.

    Features are standardized on the training split first — standard
    linear-probe practice, and what lets full-batch optimization
    actually converge inside the fixed step budget."""
    mean = train_features.mean(dim=0, keepdim=True)
    scale = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train = (train_features - mean) / scale
    held = (eval_features - mean) / scale
    torch.manual_seed(0)
    probe = nn.Linear(train.shape[1], N_CLASSES)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss_fn(probe(train), train_labels).backward()
        optimizer.step()
    with torch.no_grad():
        chosen = probe(held).argmax(dim=1)
    return float((chosen == eval_labels).float().mean())


def main() -> None:
    run_dir = Path(os.environ["ARL_RUN_DIR"])
    seed = int(os.environ.get("ARL_SEED", "0"))
    config: dict[str, object] = {}
    config_path = os.environ.get("ARL_CONFIG")
    if config_path and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text())
    dataset_root = config.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise SystemExit("this experiment needs dataset_root in config")
    smoke = bool(config.get("smoke", False))
    n_encoder = 512 if smoke else N_ENCODER_TRAIN
    n_probe_train = 512 if smoke else N_PROBE_TRAIN
    n_probe_eval = 256 if smoke else N_PROBE_EVAL
    epochs = 1 if smoke else ENCODER_EPOCHS

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    where = device(config)

    plain = torchvision.transforms.ToTensor()
    full = torchvision.datasets.CIFAR10(
        root=dataset_root, train=True, download=False, transform=plain
    )
    held_out = torchvision.datasets.CIFAR10(
        root=dataset_root, train=False, download=False, transform=plain
    )
    order = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed))
    encoder_split = Subset(full, order[:n_encoder].tolist())
    probe_split = Subset(
        full, order[n_encoder : n_encoder + n_probe_train].tolist()
    )
    eval_split = Subset(held_out, list(range(n_probe_eval)))
    tiny_split = Subset(held_out, list(range(n_probe_eval, n_probe_eval + TINY_SUBSET)))

    augment = nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, padding=4),
    )

    def fit(encoder: nn.Module, *, augmented: bool) -> float:
        with torch.no_grad():
            feature_dim = encoder(
                torch.zeros(1, 3, 32, 32, device=where)
            ).flatten(1).shape[1]
        head = nn.Linear(feature_dim, N_CLASSES).to(where)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3
        )
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(encoder_split, batch_size=BATCH, shuffle=False)
        last = 0.0
        encoder.train()
        for _ in range(epochs):
            for images, targets in loader:
                images = images.to(where)
                if augmented:
                    images = augment(images)
                optimizer.zero_grad()
                loss = loss_fn(
                    head(encoder(images).flatten(1)), targets.to(where)
                )
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
        return last

    trained = build_encoder().to(where)
    torch.manual_seed(seed + 1)  # a distinct, recorded initialization
    untrained = build_encoder().to(where)
    fit(trained, augmented=True)
    final_loss = fit(untrained, augmented=False)

    # Both arms judged identically: frozen features, the same probe.
    probe_loader = DataLoader(probe_split, batch_size=BATCH)
    eval_loader = DataLoader(eval_split, batch_size=BATCH)
    tiny_loader = DataLoader(tiny_split, batch_size=BATCH)
    scores: dict[str, float] = {}
    for name, encoder in (("augmented", trained), ("plain", untrained)):
        bank, bank_labels = features_of(encoder, probe_loader, where)
        held, held_labels = features_of(encoder, eval_loader, where)
        scores[name] = probe_top1(bank, bank_labels, held, held_labels)

    # The positive control: a probe on the trained encoder's features
    # must fit a memorizable subset scored on itself.
    tiny_bank, tiny_labels = features_of(trained, tiny_loader, where)
    overfit = probe_top1(tiny_bank, tiny_labels, tiny_bank, tiny_labels)

    metrics = {
        PRIMARY_METRIC_KEY: scores["augmented"] - scores["plain"],
        "augmented_probe_top1": scores["augmented"],
        "plain_probe_top1": scores["plain"],
        "encoder_train_loss_final": final_loss,
        "tiny_subset_overfit_top1": overfit,
        "n_probe_train": n_probe_train,
        "n_probe_eval": n_probe_eval,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
# ARL-FIXED-END contract


# ARL-SLOT-BEGIN build_encoder
def build_encoder() -> nn.Module:
    """Return a convolutional encoder for 3x32x32 CIFAR images.

    Constraints: an ``nn.Module`` whose forward maps a float tensor of
    shape (N, 3, 32, 32) to features of shape (N, D) or (N, C, H, W)
    with C*H*W at most 4096; at most two million parameters; layers from
    torch.nn only; no I/O, no randomness beyond torch's seeded default;
    do not train or load weights here — initialization only.
    """
    raise NotImplementedError("the engineer completes this slot")
# ARL-SLOT-END build_encoder


# ARL-FIXED-BEGIN entry
if __name__ == "__main__":
    main()
# ARL-FIXED-END entry
