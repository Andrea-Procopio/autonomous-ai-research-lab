"""Local, file-based persistence of research state snapshots.

``core.serialize`` renders domain objects to JSON one-way by design; parsing
them back needs validation, and that boundary work lives here rather than in
the domain core. No database — content-addressed JSON files are enough until
real volume says otherwise.
"""

from .state_store import FileStateStore, SnapshotError

__all__ = ["FileStateStore", "SnapshotError"]
