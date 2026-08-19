"""The deterministic prior-art gates: every rule a model output must
pass before anything is recorded.

Three gates, one per model-authored payload — the proposed queries, the
similarity screens, and the nearest-work comparisons — each pure: they
read the payload, the trusted source records and candidate context, and
nothing else, and they return *every* rule that fired so a corrective
call receives complete feedback. The rejection vocabulary is shared with
the mapping and ideation gates deliberately: the corrective-call
plumbing is identical, and a rule that means the same thing keeps the
same name.

Three decisions carry the epistemic weight:

* **Every reading of a prior work is held to its accessible text.** A
  dimension comparison names its support location and quotes a verbatim
  snippet; the gate re-finds the snippet in that named part of the
  source — title or abstract, exactly what was retrieved — or rejects
  the comparison. A support location the retrieval never reached
  (``full_text``, or ``abstract`` on a metadata-only source) is an
  ``access_level_mismatch``. Numbers describing the prior work must
  appear in its accessible text; numbers restating the candidate may
  also come from the candidate's own record. A known identifier is a
  name, not a number: exact ids from the rendered context are stripped
  before number extraction, while the digits of a fabricated id still
  read as ungrounded numbers.

* **The model labels; it never concludes.** A similarity label that
  contradicts its own named features — a match with no overlapping
  features, a distinction with no material differences — is a gate
  violation, and the verdict itself is computed by trusted code from
  the accepted records. Novelty language is banned outright, in both
  directions: a comparison may not call the candidate novel, and a
  screening reason may not assert that no prior work exists.

* **The cutoff is enforced, not remembered.** Trusted code already
  bounds every query and drops post-cutoff retrievals; a comparison
  citing a source dated after the cutoff is rejected here as well, so
  the discipline survives even a bug upstream.

Scientific taste is not a rule anywhere here: a thin pool, an
undecidable screen, or a comparison that finds the candidate thoroughly
anticipated passes like any other output — and has no route to a second
call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from ..ideation.gates import check_novelty_language
from ..literature.retrieval import LiteratureSource
from ..mapping.gates import (
    MappingRejection,
    accessible_text_of,
    check_coverage_language,
    check_text_integrity,
)
from .plan import (
    FORBIDDEN_TERM_CHARACTERS,
    FORBIDDEN_TERM_WORDS,
    MAX_ALTERNATIVES_PER_GROUP,
    MAX_CONCEPT_GROUPS,
    MAX_RENDERED_CHARS,
    MAX_TERM_CHARS,
    MIN_TERM_CHARS,
    canonical_groups,
    normalized_term,
    render_query,
)
from .records import (
    DIMENSIONS,
    PriorArtQueryFamily,
    SimilarityLabel,
)

# Mirrors of the mapping gates' private number helpers, kept private here
# for the same reason ideation keeps its own.
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")


def _number_tokens(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def _without_known_ids(text: str, known_ids: frozenset[str]) -> str:
    """Strip exact occurrences of known durable identifiers before
    number extraction: an id is a name, not a numerical claim, yet its
    hexadecimal tail reads as digit runs (the Task 5C.1 lesson,
    observed live 2026-08-19). Only identifiers the trusted context
    actually knows are stripped — a fabricated id's digits still
    surface as ungrounded numbers — and stripping replaces the id with
    a space, so a real number standing beside a known id is still
    validated."""
    cleaned = text
    for identifier in sorted(known_ids, key=len, reverse=True):
        cleaned = cleaned.replace(identifier, " ")
    return cleaned


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _check_prior_art_text(
    text: str, where: str, rejections: list[MappingRejection]
) -> None:
    """The checks every model-authored prior-art text passes: non-empty,
    uncorrupted, no coverage claims, no novelty claims."""
    if not text.strip():
        rejections.append(
            MappingRejection("empty_finding", f"{where} is empty")
        )
        return
    check_coverage_language(text, where, rejections)
    check_novelty_language(text, where, rejections)


def _check_grounded_numbers(
    text: str,
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
    where: str,
    rejections: list[MappingRejection],
) -> None:
    for token in sorted(_number_tokens(_without_known_ids(text, known_ids))):
        if token not in haystack_tokens:
            rejections.append(
                MappingRejection(
                    "ungrounded_number",
                    f"{where} asserts the number {token!r}, which does not "
                    f"appear in the source's accessible text or the "
                    f"candidate's own record",
                )
            )


def check_prior_art_queries(
    payload: Mapping[str, object],
    *,
    max_queries_per_family: int,
    candidate_haystack: str,
) -> tuple[MappingRejection, ...]:
    """The query-plan gate: every family covered, every plan a bounded
    set of concept groups whose Boolean meaning is explicit. The model
    supplies plain terms only — groups are conjoined, alternatives
    within a group are alternatives, and trusted code renders the one
    expression that structure means. A final query string cannot be
    expressed here at all, so whitespace can no longer silently decide
    Boolean behavior (the Task 5D live defect). Dates never appear —
    trusted code owns them."""
    rejections: list[MappingRejection] = []
    per_family: dict[str, int] = {}
    seen_plans: set[tuple[str, tuple[tuple[str, ...], ...]]] = set()
    haystack = " ".join(candidate_haystack.casefold().split())
    for index, item in enumerate(_sequence(payload["queries"])):
        assert isinstance(item, Mapping)
        where = f"queries[{index}]"
        family = str(item["family"])
        per_family[family] = per_family.get(family, 0) + 1
        groups = _plan_groups(item)
        if not groups:
            rejections.append(
                MappingRejection(
                    "empty_finding",
                    f"{where} proposes no concept groups; a search names "
                    f"at least one concept",
                )
            )
            continue
        if len(groups) > MAX_CONCEPT_GROUPS:
            rejections.append(
                MappingRejection(
                    "excessive_conjunctivity",
                    f"{where} conjoins {len(groups)} concept groups; at "
                    f"most {MAX_CONCEPT_GROUPS} are allowed — every group "
                    f"is a simultaneous requirement, and over-conjoined "
                    f"searches return nothing (the Task 5D live failure); "
                    f"drop groups or fold terms into one group's "
                    f"alternatives",
                )
            )
        plan_ok = _check_groups(groups, where, rejections)
        if not plan_ok:
            continue
        rendered = render_query(groups)
        if len(rendered) > MAX_RENDERED_CHARS:
            rejections.append(
                MappingRejection(
                    "budget_violation",
                    f"{where} renders to {len(rendered)} characters; the "
                    f"bound is {MAX_RENDERED_CHARS} — use fewer or "
                    f"shorter alternatives",
                )
            )
        anchored = any(
            normalized_term(term) in haystack
            for group in groups
            for term in group
        )
        if not anchored:
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{where} has no candidate-specific anchor: no "
                    f"alternative appears in the candidate's own record; "
                    f"ground at least one group in the candidate's terms",
                )
            )
        key = (family, canonical_groups(groups))
        if key in seen_plans:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} duplicates an earlier plan in its family",
                )
            )
        seen_plans.add(key)
    for family in PriorArtQueryFamily:
        count = per_family.get(family.value, 0)
        if count == 0:
            rejections.append(
                MappingRejection(
                    "missing_family",
                    f"no query proposed for the {family.value} family; "
                    f"every family must be searched",
                )
            )
        elif count > max_queries_per_family:
            rejections.append(
                MappingRejection(
                    "budget_violation",
                    f"the {family.value} family proposes {count} queries; "
                    f"the directive allows {max_queries_per_family}",
                )
            )
    return tuple(rejections)


def _plan_groups(item: Mapping[str, object]) -> tuple[tuple[str, ...], ...]:
    groups = []
    for group in _sequence(item["groups"]):
        assert isinstance(group, Mapping)
        groups.append(
            tuple(str(term) for term in _sequence(group["alternatives"]))
        )
    return tuple(groups)


def _check_groups(
    groups: tuple[tuple[str, ...], ...],
    where: str,
    rejections: list[MappingRejection],
) -> bool:
    """Every term a plain, bounded, uncorrupted phrase; every group a
    non-empty set of distinct alternatives; no group repeated. Returns
    False when the plan is too malformed to render meaningfully."""
    ok = True
    seen_groups: set[tuple[str, ...]] = set()
    for group_index, group in enumerate(groups):
        group_where = f"{where}.groups[{group_index}]"
        if not group:
            rejections.append(
                MappingRejection(
                    "empty_finding",
                    f"{group_where} offers no alternatives; a concept "
                    f"group names at least one term",
                )
            )
            ok = False
            continue
        if len(group) > MAX_ALTERNATIVES_PER_GROUP:
            rejections.append(
                MappingRejection(
                    "budget_violation",
                    f"{group_where} offers {len(group)} alternatives; at "
                    f"most {MAX_ALTERNATIVES_PER_GROUP} are allowed — "
                    f"alternatives are synonyms for one concept, not a "
                    f"topic list",
                )
            )
        seen_terms: set[str] = set()
        for term_index, term in enumerate(group):
            term_where = f"{group_where}.alternatives[{term_index}]"
            if not term.strip():
                rejections.append(
                    MappingRejection(
                        "empty_finding", f"{term_where} is empty"
                    )
                )
                ok = False
                continue
            check_text_integrity(term, term_where, rejections)
            stripped = term.strip()
            if not (MIN_TERM_CHARS <= len(stripped) <= MAX_TERM_CHARS):
                rejections.append(
                    MappingRejection(
                        "malformed_finding",
                        f"{term_where} must be {MIN_TERM_CHARS}.."
                        f"{MAX_TERM_CHARS} characters of plain search "
                        f"text, got {len(stripped)}",
                    )
                )
            bad_characters = sorted(
                set(stripped) & FORBIDDEN_TERM_CHARACTERS
            )
            bad_words = sorted(
                {
                    word
                    for word in stripped.casefold().split()
                    if word in FORBIDDEN_TERM_WORDS
                }
            )
            if bad_characters or bad_words:
                offending = ", ".join(
                    (*map(repr, bad_characters), *map(repr, bad_words))
                )
                rejections.append(
                    MappingRejection(
                        "unsupported_syntax",
                        f"{term_where} carries Boolean, wildcard, or "
                        f"quoting syntax ({offending}); supply plain "
                        f"terms — structure comes from the groups, and "
                        f"trusted code renders the operators",
                    )
                )
            key = normalized_term(term)
            if key in seen_terms:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{term_where} duplicates an earlier alternative "
                        f"in its group",
                    )
                )
            seen_terms.add(key)
        group_key = tuple(sorted(seen_terms))
        if group_key and group_key in seen_groups:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{group_where} repeats an earlier concept group",
                )
            )
        seen_groups.add(group_key)
    return ok


def check_similarity_screening(
    payload: Mapping[str, object],
    *,
    accessible: Mapping[str, str],
    candidate_tokens: frozenset[str],
    known_ids: frozenset[str],
) -> tuple[MappingRejection, ...]:
    """The screening gate. ``accessible`` maps each rendered source id
    to its accessible text — together the only universe a screen may
    judge. Every rendered source must be decided exactly once; every
    reason is held to the full text discipline, with numbers grounded
    in the source's accessible text or the candidate's own record."""
    rejections: list[MappingRejection] = []
    decided: set[str] = set()
    for index, item in enumerate(_sequence(payload["screens"])):
        assert isinstance(item, Mapping)
        where = f"screens[{index}]"
        source_id = str(item["source_id"])
        reason = str(item["reason"])
        if source_id not in accessible:
            rejections.append(
                MappingRejection(
                    "unknown_source",
                    f"{where} screens {source_id!r}, which was not among "
                    f"the rendered sources; screen only the listed ids",
                )
            )
            continue
        if source_id in decided:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} screens {source_id} a second time",
                )
            )
            continue
        decided.add(source_id)
        _check_prior_art_text(reason, f"{where}.reason", rejections)
        _check_grounded_numbers(
            reason,
            _number_tokens(accessible[source_id]) | candidate_tokens,
            known_ids,
            f"{where}.reason",
            rejections,
        )
    for source_id in accessible:
        if source_id not in decided:
            rejections.append(
                MappingRejection(
                    "missing_decision",
                    f"source {source_id} was rendered but never screened; "
                    f"every listed source gets exactly one decision",
                )
            )
    return tuple(rejections)


def check_comparisons(
    payload: Mapping[str, object],
    *,
    sources: Mapping[str, LiteratureSource],
    cutoff_date: str,
    candidate_tokens: frozenset[str],
    known_ids: frozenset[str],
) -> tuple[MappingRejection, ...]:
    """The comparison gate. ``sources`` maps each rendered work's id to
    its snapshot record — together the only universe a comparison may
    read, so an excluded or metadata-only source is unknown here by
    construction. Every rendered work must be compared exactly once,
    across all five dimensions, each grounded in a verbatim snippet of
    the named accessible part."""
    rejections: list[MappingRejection] = []
    compared: set[str] = set()
    entries = tuple(_sequence(payload["comparisons"]))
    if len(entries) > len(sources):
        rejections.append(
            MappingRejection(
                "budget_violation",
                f"{len(entries)} comparisons over {len(sources)} rendered "
                f"works; compare each listed work exactly once",
            )
        )
    for index, item in enumerate(entries):
        assert isinstance(item, Mapping)
        where = f"comparisons[{index}]"
        source_id = str(item["source_id"])
        if source_id not in sources:
            rejections.append(
                MappingRejection(
                    "unknown_source",
                    f"{where} compares {source_id!r}, which was not among "
                    f"the rendered works; compare only the listed ids",
                )
            )
            continue
        if source_id in compared:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} compares {source_id} a second time",
                )
            )
            continue
        compared.add(source_id)
        source = sources[source_id]
        if (
            cutoff_date
            and source.publication_date is not None
            and source.publication_date > cutoff_date
        ):
            rejections.append(
                MappingRejection(
                    "post_cutoff_source",
                    f"{where} compares {source_id}, dated "
                    f"{source.publication_date}, after the recorded "
                    f"cutoff {cutoff_date}",
                )
            )
        _check_comparison_body(
            item,
            where,
            source=source,
            candidate_tokens=candidate_tokens,
            known_ids=known_ids,
            rejections=rejections,
        )
    for source_id in sources:
        if source_id not in compared:
            rejections.append(
                MappingRejection(
                    "missing_decision",
                    f"work {source_id} was rendered but never compared; "
                    f"every listed work gets exactly one comparison",
                )
            )
    return tuple(rejections)


def _check_comparison_body(
    item: Mapping[str, object],
    where: str,
    *,
    source: LiteratureSource,
    candidate_tokens: frozenset[str],
    known_ids: frozenset[str],
    rejections: list[MappingRejection],
) -> None:
    source_tokens = _number_tokens(accessible_text_of(source))
    similarity = str(item["similarity"])
    overlap = tuple(
        str(entry) for entry in _sequence(item["overlap_features"])
    )
    differences = tuple(
        str(entry) for entry in _sequence(item["material_differences"])
    )
    if similarity != SimilarityLabel.DISTINCT.value and not overlap:
        rejections.append(
            MappingRejection(
                "ungrounded_overlap",
                f"{where} labels the work {similarity} yet names no "
                f"overlapping features; a similarity claim rests on named "
                f"features",
            )
        )
    if similarity != SimilarityLabel.SUBSTANTIAL_MATCH.value and (
        not differences
    ):
        rejections.append(
            MappingRejection(
                "similarity_contradiction",
                f"{where} labels the work {similarity} yet names no "
                f"material differences; the label contradicts its own "
                f"account",
            )
        )
    for name, items, haystack_tokens in (
        ("overlap_features", overlap, source_tokens),
        (
            "material_differences",
            differences,
            source_tokens | candidate_tokens,
        ),
    ):
        seen: set[str] = set()
        for index, text in enumerate(items):
            entry_where = f"{where}.{name}[{index}]"
            _check_prior_art_text(text, entry_where, rejections)
            _check_grounded_numbers(
                text, haystack_tokens, known_ids, entry_where, rejections
            )
            key = _normalized(text)
            if key and key in seen:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{entry_where} duplicates an earlier entry",
                    )
                )
            seen.add(key)
    covered: set[str] = set()
    for index, entry in enumerate(_sequence(item["dimensions"])):
        assert isinstance(entry, Mapping)
        entry_where = f"{where}.dimensions[{index}]"
        dimension = str(entry["dimension"])
        if dimension in covered:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{entry_where} covers {dimension} a second time",
                )
            )
        covered.add(dimension)
        candidate_position = str(entry["candidate_position"])
        prior_position = str(entry["prior_work_position"])
        _check_prior_art_text(
            candidate_position, f"{entry_where}.candidate_position", rejections
        )
        _check_prior_art_text(
            prior_position, f"{entry_where}.prior_work_position", rejections
        )
        _check_grounded_numbers(
            candidate_position,
            source_tokens | candidate_tokens,
            known_ids,
            f"{entry_where}.candidate_position",
            rejections,
        )
        _check_grounded_numbers(
            prior_position,
            source_tokens,
            known_ids,
            f"{entry_where}.prior_work_position",
            rejections,
        )
        _check_support(
            entry, entry_where, source=source, rejections=rejections
        )
    missing = sorted(
        dimension.value
        for dimension in DIMENSIONS
        if dimension.value not in covered
    )
    if missing:
        rejections.append(
            MappingRejection(
                "missing_dimension",
                f"{where} skips the {', '.join(missing)} "
                f"dimension{'s' if len(missing) > 1 else ''}; every "
                f"comparison covers all five",
            )
        )


def _check_support(
    entry: Mapping[str, object],
    where: str,
    *,
    source: LiteratureSource,
    rejections: list[MappingRejection],
) -> None:
    """The snippet must appear verbatim (whitespace-normalized,
    case-insensitive) in the part of the source the location names —
    a reading of text that was never retrieved is a mismatch, not a
    style issue."""
    location = str(entry["support_location"])
    snippet = str(entry["support_snippet"])
    if not snippet.strip():
        rejections.append(
            MappingRejection(
                "missing_support",
                f"{where} carries no support snippet; quote the accessible "
                f"text the reading rests on",
            )
        )
        return
    check_text_integrity(snippet, f"{where}.support_snippet", rejections)
    if location == "title":
        haystack = source.title or ""
    elif location == "abstract":
        if source.abstract is None:
            rejections.append(
                MappingRejection(
                    "access_level_mismatch",
                    f"{where} claims abstract support for {source.id}, "
                    f"whose abstract was not retrieved",
                )
            )
            return
        haystack = source.abstract
    else:
        rejections.append(
            MappingRejection(
                "access_level_mismatch",
                f"{where} claims {location!r} support; only title and "
                f"abstract text was retrieved",
            )
        )
        return
    if _normalized(snippet) not in _normalized(haystack):
        rejections.append(
            MappingRejection(
                "unsupported_claim",
                f"{where} quotes {snippet!r}, which does not appear in the "
                f"source's {location}; quote the {location} verbatim",
            )
        )


def _sequence(value: object) -> Sequence[object]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return value
