"""Verifying a whole run from cold.

One deterministic pass over a run root, run by a process that wrote none
of it. It answers a single question — *is everything this run claims
still here, and still what it said it was?* — and answers it with typed
issues, never with verdicts. What a broken run means is somebody else's
call; this module only says what is broken.

Six checks, each reusing the guarantee the writing layer already built:

============  ============================================================
snapshots     ``FileStateStore.load`` recomputes each state's content id
records       every result and evidence payload re-hashes to its digest
references    every ``ResultRef`` and evidence id in every state resolves
artifacts     every manifest entry has a blob that still hashes to it
chain         ``validate_evidence_chain`` on each leaf state
ledger        the funded run replays to the balance its state carries
============  ============================================================

The chain check runs on leaf states — the ones nothing else descends
from — rather than on every snapshot: a lineage's intermediate states
are already covered by the leaf that grew out of them, and running the
full chain check on all of them turns a linear pass into a quadratic
one.

Nothing here raises for a broken run. A missing blob, a corrupt payload,
a state citing a fact nobody stored — each becomes an issue in the
report, because the point of a verifier is to survive what it finds and
say all of it at once.

It lives in ``program`` rather than ``evidence`` because of what it has
to reach: snapshots from ``persistence``, facts and artifacts from
``evidence``, and the budget ledger from here. ``evidence`` depends on
``core`` alone and should keep doing so; the package that owns what a
run *is* is the one that can say whether a run is intact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..core.state import ResearchState
from ..evidence.artifacts import ArtifactEntry, ArtifactManifest
from ..evidence.file_store import FileEvidenceStore
from ..evidence.validation import validate_evidence_chain
from ..persistence.state_store import FileStateStore, SnapshotError
from .store import ProgramStore


class IntegrityIssueKind(StrEnum):
    UNREADABLE_SNAPSHOT = "unreadable_snapshot"
    """A state snapshot is missing, malformed, or no longer hashes to its
    own filename."""

    UNREADABLE_RECORD = "unreadable_record"
    """A result or evidence payload does not survive its own digest."""

    MISSING_FACT = "missing_fact"
    """A state references a result or evidence the store does not hold."""

    MISSING_MANIFEST = "missing_manifest"
    """A recorded result that named outputs has no artifact manifest."""

    MISSING_BLOB = "missing_blob"
    """A manifest names bytes the blob store does not hold."""

    CORRUPT_BLOB = "corrupt_blob"
    """Stored bytes no longer hash to the digest they are filed under."""

    CHAIN_ISSUE = "chain_issue"
    """The evidence chain reports a problem — carried through with its own
    kind in the detail, not re-judged here."""

    LEDGER_ISSUE = "ledger_issue"
    """A funded run's envelope, state, or budget ledger does not reload
    and reconcile."""


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    kind: IntegrityIssueKind
    subject_id: str
    detail: str


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """What was checked, and everything found wrong."""

    root: str
    states_checked: int
    results_checked: int
    evidence_checked: int
    blobs_checked: int
    issues: tuple[IntegrityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def of_kind(
        self, kind: IntegrityIssueKind
    ) -> tuple[IntegrityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.kind is kind)


def verify_run(
    root: Path | str, *, program_root: Path | None = None
) -> IntegrityReport:
    """Verify every durable claim under ``root``.

    ``program_root`` names a funded run's records when they live
    somewhere other than ``<root>/program``. Absent both, the ledger
    check is skipped — an unfunded run root is a complete run root for
    every other purpose.
    """
    root = Path(root)
    issues: list[IntegrityIssue] = []
    states = _load_states(root, issues)
    store = FileEvidenceStore(root)
    results = _check_records(store, issues)
    evidence_count = _check_evidence(store, issues)
    blobs = _check_artifacts(store, results, issues)
    _check_references(states, store, issues)
    _check_chain(states, store, issues)
    _check_ledger(root, program_root, states, issues)
    return IntegrityReport(
        root=str(root),
        states_checked=len(states),
        results_checked=len(results),
        evidence_checked=evidence_count,
        blobs_checked=blobs,
        issues=tuple(issues),
    )


# -- checks --------------------------------------------------------------------


def _load_states(
    root: Path, issues: list[IntegrityIssue]
) -> list[ResearchState]:
    store = FileStateStore(root)
    loaded: list[ResearchState] = []
    for state_id in store.state_ids():
        try:
            loaded.append(store.load(state_id))
        except SnapshotError as error:
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.UNREADABLE_SNAPSHOT,
                    subject_id=state_id,
                    detail=str(error),
                )
            )
    return loaded


def _check_records(
    store: FileEvidenceStore, issues: list[IntegrityIssue]
) -> list[str]:
    readable: list[str] = []
    for result_id in store.result_ids():
        try:
            store.get_result(result_id)
        except Exception as error:  # reported, never re-raised
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.UNREADABLE_RECORD,
                    subject_id=result_id,
                    detail=str(error),
                )
            )
            continue
        readable.append(result_id)
    return readable


def _check_evidence(
    store: FileEvidenceStore, issues: list[IntegrityIssue]
) -> int:
    readable = 0
    for evidence_id in store.evidence_ids():
        try:
            store.get_evidence(evidence_id)
        except Exception as error:  # reported, never re-raised
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.UNREADABLE_RECORD,
                    subject_id=evidence_id,
                    detail=str(error),
                )
            )
            continue
        readable += 1
    return readable


def _check_artifacts(
    store: FileEvidenceStore,
    result_ids: list[str],
    issues: list[IntegrityIssue],
) -> int:
    checked = 0
    for result_id in result_ids:
        result = store.get_result(result_id)
        try:
            manifest = store.artifacts.get(result_id)
        except Exception as error:  # reported, never re-raised
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.UNREADABLE_RECORD,
                    subject_id=result_id,
                    detail=f"artifact manifest: {error}",
                )
            )
            continue
        if manifest is None:
            if result.artifacts or result.logs:
                issues.append(
                    IntegrityIssue(
                        kind=IntegrityIssueKind.MISSING_MANIFEST,
                        subject_id=result_id,
                        detail=(
                            f"result names "
                            f"{len(result.artifacts) + len(result.logs)} "
                            f"file(s) but no manifest was stored"
                        ),
                    )
                )
            continue
        checked += _check_blobs(store, manifest, issues)
    return checked


def _check_blobs(
    store: FileEvidenceStore,
    manifest: ArtifactManifest,
    issues: list[IntegrityIssue],
) -> int:
    checked = 0
    for entry in manifest.entries:
        path = store.artifacts.blob_path(entry.digest)
        if not path.is_file():
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.MISSING_BLOB,
                    subject_id=manifest.result_id,
                    detail=_where(entry, "is not in the blob store"),
                )
            )
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.digest:
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.CORRUPT_BLOB,
                    subject_id=manifest.result_id,
                    detail=_where(entry, "no longer hashes to its digest"),
                )
            )
            continue
        checked += 1
    return checked


def _check_references(
    states: list[ResearchState],
    store: FileEvidenceStore,
    issues: list[IntegrityIssue],
) -> None:
    known_results = set(store.result_ids())
    known_evidence = set(store.evidence_ids())
    for state in states:
        for ref in state.results:
            if ref.result_id not in known_results:
                issues.append(
                    IntegrityIssue(
                        kind=IntegrityIssueKind.MISSING_FACT,
                        subject_id=state.id,
                        detail=(
                            f"state references result {ref.result_id}, which "
                            f"this store does not hold"
                        ),
                    )
                )
        for evidence_id in state.evidence_ids:
            if evidence_id not in known_evidence:
                issues.append(
                    IntegrityIssue(
                        kind=IntegrityIssueKind.MISSING_FACT,
                        subject_id=state.id,
                        detail=(
                            f"state references evidence {evidence_id}, which "
                            f"this store does not hold"
                        ),
                    )
                )


def _check_chain(
    states: list[ResearchState],
    store: FileEvidenceStore,
    issues: list[IntegrityIssue],
) -> None:
    for state in _leaves(states):
        try:
            found = validate_evidence_chain(state, store)
        except Exception as error:  # reported, never re-raised
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.CHAIN_ISSUE,
                    subject_id=state.id,
                    detail=f"the chain could not be walked: {error}",
                )
            )
            continue
        issues.extend(
            IntegrityIssue(
                kind=IntegrityIssueKind.CHAIN_ISSUE,
                subject_id=issue.subject_id,
                detail=f"{issue.kind}: {issue.detail}",
            )
            for issue in found
        )


def _check_ledger(
    root: Path,
    program_root: Path | None,
    states: list[ResearchState],
    issues: list[IntegrityIssue],
) -> None:
    """The ledger must agree with a state the run actually reached.

    Which state depends on how far the run has got. An unspent run's
    balance is its grant, and equals the funded snapshot's budget
    because that snapshot *is* the grant. A run that has spent agrees
    with neither: the funded snapshot is immutable and keeps the grant
    forever, while the balance tracks whatever the run has committed
    since. Comparing against the funded snapshot alone reports every run
    that did any work, which is the wrong way round.

    The check is therefore "some snapshot under this root holds this
    balance" rather than "the head of this run's chain does", and that
    is a real weakening. It is forced: the snapshots a run leaves are a
    sequence of committed heads, not a linked list. One step evolves the
    state several times — begin the attempt, apply the transition,
    resolve it — and persists only what it committed, so each snapshot's
    parent is an intermediate that was never written down. Recovering
    the order would need the runtime to persist every evolution, which
    is a cost decision and not this pass's to make.
    """
    resolved = program_root if program_root is not None else root / "program"
    if not (resolved / "envelopes").is_dir():
        return
    store = ProgramStore(resolved)
    try:
        envelopes = store.runs()
    except Exception as error:  # reported, never re-raised
        issues.append(
            IntegrityIssue(
                kind=IntegrityIssueKind.LEDGER_ISSUE,
                subject_id=str(resolved),
                detail=f"the run envelopes could not be listed: {error}",
            )
        )
        return
    for envelope in envelopes:
        try:
            run, state = store.get_funded_state(envelope.id)
            balance = store.ledger_for(run.run_id).balance()
        except Exception as error:  # reported, never re-raised
            issues.append(
                IntegrityIssue(
                    kind=IntegrityIssueKind.LEDGER_ISSUE,
                    subject_id=envelope.run_id,
                    detail=str(error),
                )
            )
            continue
        reached = {snapshot.budget for snapshot in states}
        if balance in {run.granted, state.budget, *reached}:
            continue
        issues.append(
            IntegrityIssue(
                kind=IntegrityIssueKind.LEDGER_ISSUE,
                subject_id=run.run_id,
                detail=(
                    f"the ledger replays to {balance}, which is neither the "
                    f"grant {run.granted}, nor the funded state's "
                    f"{state.budget}, nor the budget of any of the "
                    f"{len(states)} snapshot(s) this root holds"
                ),
            )
        )


# -- helpers -------------------------------------------------------------------


def _leaves(states: list[ResearchState]) -> list[ResearchState]:
    """States nothing else descends from. An intermediate state's chain is
    already covered by the leaf that grew out of it."""
    parents = {state.parent_id for state in states if state.parent_id}
    return [state for state in states if state.id not in parents]


def _where(entry: ArtifactEntry, problem: str) -> str:
    return f"{entry.path} ({entry.digest[:12]}…) {problem}"
