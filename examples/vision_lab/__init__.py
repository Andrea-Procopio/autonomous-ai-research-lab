"""The production vision lab: real training over a funded run.

The first lab (``--lab examples.vision_lab:...``) whose experiments are
actual machine learning — CIFAR-10-scale representation learning, minutes
per run — behind the same contracts every canary walk proved: the
directive-driven chain, the attempt journal, the budget ledger, the
deterministic gates.

What lives where:

``datasets``
    Content-addressed dataset manifests beside operator-staged bytes, a
    write-once store, and the preflight check that refuses a job whose
    dataset does not verify.
``backends``
    The one place execution backends differ: a deployment profile chooses
    host or container, CPU or GPU, and resolves to the existing
    ``JobBinding``/``Executor`` seams. Nothing scientific varies with it.
``measure`` / ``catalog`` / ``science`` / ``composition``
    The lab itself: the metric grammar and its honest refusal, the trusted
    template catalog, the deterministic science roles, and the ``Lab``
    factories the CLI loads.

Torch stays out of this package's imports deliberately: templates are
source-as-data completed by the engineer's model and executed by a
backend; the lab machinery itself must typecheck and test on a machine
with no ML stack installed.
"""

from .composition import (
    VisionLab,
    ci_lab,
    lab,
    qualification_lab,
)

__all__ = ["VisionLab", "ci_lab", "lab", "qualification_lab"]
