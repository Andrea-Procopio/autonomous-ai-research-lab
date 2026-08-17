"""Identity semantics: two kinds of identifier for two kinds of question.

**Semantic (content) identity** — :func:`content_id` — answers "is this the
same scientific object?" The id is a hash of the object's content, so the same
hypothesis constructed twice — in a replay, a re-run, or a parallel branch of a
research search — carries the same id, and trajectories from different runs can
be compared object by object. Used for immutable semantic objects: hypotheses,
predictions, experiment specs, claims, evidence readings.

**Occurrence (event) identity** — :func:`occurrence_id` — answers "is this the
same event?" Two identical experiment executions are different events even
though they share an `ExperimentSpec`; a retry of an action is a different
attempt even though it carries the same intent. Occurrence ids are unique per
construction, never derived from content. Used for execution events: action
attempts, experiment jobs, decisions.

The invariant, tested in ``tests/test_identity.py``:

    identical content        -> identical content id
    identical construction   -> distinct occurrence ids

Confusing the two corrupts provenance in one of two ways: content ids on events
collapse retries into one record; occurrence ids on semantic objects make the
same hypothesis look like a different one on every branch.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Final

from .types import canonical

_DIGEST_BYTES: Final = 8
_SEPARATOR: Final = "\x1f"


def content_id(prefix: str, *parts: object) -> str:
    """Return ``<prefix>_<16 hex chars>`` deterministically derived from ``parts``."""
    payload = _SEPARATOR.join(canonical(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES)
    return f"{prefix}_{digest.hexdigest()}"


def occurrence_id(prefix: str) -> str:
    """Return ``<prefix>_<16 hex chars>``, unique per call."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"
