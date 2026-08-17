"""Deterministic, content-addressed identifiers.

Identity is derived from content rather than from a counter or a clock. The
same scientific object constructed twice -- in a replay, in a re-run, or on a
parallel branch of a research search tree -- therefore carries the same id, so
provenance chains stay comparable across executions.

Objects that must stay distinct across otherwise-identical constructions (a
replication of the same experiment, for example) include an explicit
discriminator in their content.
"""

from __future__ import annotations

import hashlib
from typing import Final

from .types import canonical

_DIGEST_BYTES: Final = 8
_SEPARATOR: Final = "\x1f"


def content_id(prefix: str, *parts: object) -> str:
    """Return ``<prefix>_<16 hex chars>`` derived from ``parts``."""
    payload = _SEPARATOR.join(canonical(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES)
    return f"{prefix}_{digest.hexdigest()}"
