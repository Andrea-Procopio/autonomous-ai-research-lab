"""Append-only storage of experimental facts, and the derived chain checker.

Two implementations of the same contract: :class:`InMemoryEvidenceStore`
for tests and explicit ablations, and :class:`FileEvidenceStore` for a
run whose facts must outlive the process that produced them — with
:class:`FileArtifactStore` keeping the bytes those facts point at.
"""

from .artifacts import (
    MANIFEST_FILENAME,
    MAX_BLOB_BYTES,
    ArtifactConflictError,
    ArtifactEntry,
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactManifest,
    ArtifactRefusedError,
    ArtifactStore,
    FileArtifactStore,
)
from .file_store import EvidenceIntegrityError, FileEvidenceStore
from .store import (
    EvidenceConflictError,
    EvidenceStore,
    InMemoryEvidenceStore,
    UnknownRecordError,
)
from .validation import ChainIssue, ChainIssueKind, validate_evidence_chain

__all__ = [
    "MANIFEST_FILENAME",
    "MAX_BLOB_BYTES",
    "ArtifactConflictError",
    "ArtifactEntry",
    "ArtifactIntegrityError",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactRefusedError",
    "ArtifactStore",
    "ChainIssue",
    "ChainIssueKind",
    "EvidenceConflictError",
    "EvidenceIntegrityError",
    "EvidenceStore",
    "FileArtifactStore",
    "FileEvidenceStore",
    "InMemoryEvidenceStore",
    "UnknownRecordError",
    "validate_evidence_chain",
]
