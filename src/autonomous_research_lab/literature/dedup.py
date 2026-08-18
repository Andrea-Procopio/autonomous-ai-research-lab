"""Conservative, deterministic deduplication over literature snapshots.

Source records are snapshots — the same paper retrieved twice, or reported
by two routes, is two records. This module answers the work-level question
("which snapshots describe the same paper?") without ever rewriting or
discarding a record: the output is a *report* — groups of duplicate
snapshots, one representative each, and every conflict that prevented a
merge — never a mutated corpus.

The rules, in order of trust:

1. **Exact canonical identifiers.** Snapshots sharing a normalized DOI, a
   normalized arXiv id, or the same provider's own work id are the same
   work. These identifiers exist to name works; agreement on any one of
   them is sufficient.
2. **Title fallback, only where it is safe.** A snapshot with *no*
   canonical identifier at all may be matched by the triple (normalized
   title, publication year, first author's family name) — all three
   required, and only against other identifier-less snapshots. Where a
   canonical identifier exists, it alone decides; title similarity never
   overrides or supplements it.
3. **Conflicts are never silently merged.** A candidate merge that would
   put two *different* DOIs, or two different arXiv ids, into one group —
   the same arXiv id under two DOIs, a provider id whose snapshots
   disagree about their DOI — is refused, kept as separate records, and
   reported as a :class:`DuplicateConflict`. Contradictory identity is a
   finding, not a tie to break.

Everything is deterministic in the input order: same snapshots in the same
order, same report, no randomness, no similarity scores.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .retrieval import LiteratureSource

MIN_TITLE_KEY_CHARS: Final = 10
"""A normalized title shorter than this is too generic to be identity
evidence — 'introduction' and kin must never fuse distinct works."""

_NON_ALNUM: Final = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Two or more snapshots judged to describe one work. ``source_ids``
    keeps input order; the first entry is the group's representative."""

    source_ids: tuple[str, ...]
    matched_on: tuple[str, ...]
    """The kinds of key that united the group (``doi``, ``arxiv``,
    ``provider_id``, ``title``), sorted."""


@dataclass(frozen=True, slots=True)
class DuplicateConflict:
    """A refused merge: these snapshots matched on ``kind``/``key`` but
    contradict each other on another canonical identifier. They remain
    separate records; the contradiction itself is the finding."""

    kind: str
    key: str
    source_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    total: int
    representative_ids: tuple[str, ...]
    """One id per distinct work, in first-appearance order."""

    groups: tuple[DuplicateGroup, ...]
    conflicts: tuple[DuplicateConflict, ...]

    @property
    def duplicate_count(self) -> int:
        return self.total - len(self.representative_ids)


def deduplicate(sources: Sequence[LiteratureSource]) -> DeduplicationReport:
    """Group ``sources`` into distinct works, conservatively.

    Deterministic union-find in input order. Each source contributes its
    canonical keys; two sources sharing a key merge unless the merged
    group would contain contradictory canonical identifiers, in which
    case the merge is refused and reported.
    """
    parent = list(range(len(sources)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    dois: dict[int, set[str]] = {}
    arxivs: dict[int, set[str]] = {}
    kinds: dict[int, set[str]] = {}
    for index, source in enumerate(sources):
        dois[index] = {source.doi} if source.doi is not None else set()
        arxivs[index] = {source.arxiv_id} if source.arxiv_id is not None else set()
        kinds[index] = set()

    conflicts: list[DuplicateConflict] = []
    key_owner: dict[tuple[str, str], int] = {}
    for index, source in enumerate(sources):
        for kind, key in _keys(source):
            owner = key_owner.setdefault((kind, key), index)
            if owner == index:
                continue
            left, right = find(owner), find(index)
            if left == right:
                kinds[left].add(kind)
                continue
            merged_dois = dois[left] | dois[right]
            merged_arxivs = arxivs[left] | arxivs[right]
            if len(merged_dois) > 1 or len(merged_arxivs) > 1:
                conflicts.append(
                    _conflict(
                        sources, kind, key, left, right, merged_dois, merged_arxivs
                    )
                )
                continue
            # The smaller index stays the root, so representatives follow
            # first appearance no matter the merge order.
            root, absorbed = min(left, right), max(left, right)
            parent[absorbed] = root
            dois[root] = merged_dois
            arxivs[root] = merged_arxivs
            kinds[root] |= kinds[absorbed] | {kind}

    members: dict[int, list[int]] = {}
    for index in range(len(sources)):
        members.setdefault(find(index), []).append(index)

    representative_ids = []
    groups = []
    for root in sorted(members):
        component = members[root]
        representative_ids.append(sources[component[0]].id)
        if len(component) > 1:
            groups.append(
                DuplicateGroup(
                    source_ids=tuple(sources[i].id for i in component),
                    matched_on=tuple(sorted(kinds[root])),
                )
            )
    return DeduplicationReport(
        total=len(sources),
        representative_ids=tuple(representative_ids),
        groups=tuple(groups),
        conflicts=tuple(conflicts),
    )


def _keys(source: LiteratureSource) -> tuple[tuple[str, str], ...]:
    """The identity keys one snapshot contributes, strongest first."""
    keys = [("provider_id", f"{source.provider}:{source.provider_id}")]
    if source.doi is not None:
        keys.append(("doi", source.doi))
    if source.arxiv_id is not None:
        keys.append(("arxiv", source.arxiv_id))
    if source.doi is None and source.arxiv_id is None:
        title_key = _title_key(source)
        if title_key is not None:
            keys.append(("title", title_key))
    return tuple(keys)


def _title_key(source: LiteratureSource) -> str | None:
    """The safe-fallback key, or ``None`` when any leg is missing: a
    normalized title alone is not identity evidence."""
    if (
        source.title is None
        or source.publication_year is None
        or not source.authors
    ):
        return None
    title = _normalize_text(source.title)
    surname = _family_name(source.authors[0])
    if len(title) < MIN_TITLE_KEY_CHARS or not surname:
        return None
    return f"{title}|{source.publication_year}|{surname}"


def _normalize_text(value: str) -> str:
    return _NON_ALNUM.sub(" ", value.casefold()).strip()


def _family_name(author: str) -> str:
    """The family name from either byline convention: the part before a
    comma ('Ajieh, Frank'), else the last token ('Frank Ajieh')."""
    if "," in author:
        candidate = author.split(",", 1)[0]
    else:
        tokens = author.split()
        candidate = tokens[-1] if tokens else ""
    return _normalize_text(candidate)


def _conflict(
    sources: Sequence[LiteratureSource],
    kind: str,
    key: str,
    left_root: int,
    right_root: int,
    merged_dois: set[str],
    merged_arxivs: set[str],
) -> DuplicateConflict:
    disagreements = []
    if len(merged_dois) > 1:
        disagreements.append(f"DOIs {' vs '.join(sorted(merged_dois))}")
    if len(merged_arxivs) > 1:
        disagreements.append(f"arXiv ids {' vs '.join(sorted(merged_arxivs))}")
    return DuplicateConflict(
        kind=kind,
        key=key,
        source_ids=(sources[left_root].id, sources[right_root].id),
        detail=(
            f"matched on {kind} {key!r} but carry conflicting "
            f"{'; '.join(disagreements)}"
        ),
    )
