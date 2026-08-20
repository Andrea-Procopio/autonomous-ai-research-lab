"""The fault harness, imported from the driver that demonstrates it.

The machinery lives in :mod:`examples.torn_step` rather than here for a
plain reason: the driver and the tests exercise the same mechanism, and
two copies of it would drift. This module exists so a test reads as
importing test infrastructure rather than an example.
"""

from __future__ import annotations

from examples.torn_step import (
    Faults,
    FaultyBundles,
    FaultyEvidence,
    FaultyJournal,
    FaultyLab,
    FaultyLedger,
    FaultyStates,
    SimulatedCrashError,
)

__all__ = [
    "Faults",
    "FaultyBundles",
    "FaultyEvidence",
    "FaultyJournal",
    "FaultyLab",
    "FaultyLedger",
    "FaultyStates",
    "SimulatedCrashError",
]
