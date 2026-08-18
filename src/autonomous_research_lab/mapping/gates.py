"""The deterministic field-mapping gates: every rule a model output must
pass before anything is recorded.

Four gates, one per model-authored payload — query proposal, screening
batch, per-source extraction, field map, problem inventory — each pure:
they read the payload, the brief, the trusted Task 5A sources, and
nothing else, and they return *every* rule that fired so a corrective
call receives complete feedback.

Two checks are worth naming because they carry the epistemic weight:

* **Verbatim grounding.** Every number token in a model-authored claim
  must appear as the same token in the cited sources' accessible text
  (title, abstract, and bibliographic fields), and every dataset name —
  plus any reported version, size, URL, or license — must appear
  verbatim (case-insensitive) in that text. What cannot be found in the
  accessible material cannot be recorded as reported by it. Paraphrase
  is allowed only where nothing mechanically checkable is asserted.

* **Coverage language.** No model-authored text may claim exhaustive
  coverage, a systematic review, or proven novelty. A bounded mapping
  run cannot know any of those things, so language asserting them is a
  gate violation, not a style issue.

Scientific taste is not a rule anywhere here: a map the lab finds thin,
or a problem inventory it finds disappointing, passes like any other.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..literature.retrieval import AccessLevel, LiteratureSource
from .brief import REQUIRED_FAMILIES, ResearchBrief, SourceEra
from .records import SupportLocation, ThemeEra

MAX_QUERY_TEXT_CHARS: Final = 300

#: Case-insensitive phrases no model-authored text may contain. A bounded
#: run retrieved a slice; these words claim the whole.
BANNED_COVERAGE_PHRASES: Final = (
    "exhaustive",
    "systematic review",
    "all relevant work",
    "all prior work",
    "all existing work",
    "complete coverage",
    "fully comprehensive",
    "proven novel",
    "provably novel",
    "proven novelty",
    "guaranteed novel",
    "first ever",
    "never been studied",
    "no prior work exists",
)

_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class MappingRejection:
    """One deterministic gate rule that fired, with its specifics."""

    rule: str
    detail: str


def accessible_text_of(source: LiteratureSource) -> str:
    """Everything Task 5A actually retrieved for one source, as one
    case-folded haystack: the bibliographic facts plus the abstract when
    the access level includes one. This — and nothing else — is what a
    claim about the source may be grounded in."""
    parts = [
        source.title or "",
        source.abstract or "",
        source.publication_date or "",
        str(source.publication_year or ""),
        source.venue or "",
        source.work_type or "",
        *source.authors,
        source.doi or "",
        source.arxiv_id or "",
    ]
    return "\n".join(part for part in parts if part).casefold()


def _number_tokens(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def check_coverage_language(
    text: str, where: str, rejections: list[MappingRejection]
) -> None:
    lowered = text.casefold()
    for phrase in BANNED_COVERAGE_PHRASES:
        if phrase in lowered:
            rejections.append(
                MappingRejection(
                    "coverage_language",
                    f"{where} claims what a bounded run cannot know "
                    f"({phrase!r}): {text!r}",
                )
            )


def _check_grounded_numbers(
    text: str,
    haystack_tokens: frozenset[str],
    where: str,
    rejections: list[MappingRejection],
) -> None:
    for token in sorted(_number_tokens(text)):
        if token not in haystack_tokens:
            rejections.append(
                MappingRejection(
                    "ungrounded_number",
                    f"{where} asserts the number {token!r}, which does not "
                    f"appear in the cited sources' accessible text",
                )
            )


def _check_texts(
    items: Sequence[str],
    where: str,
    haystack_tokens: frozenset[str],
    rejections: list[MappingRejection],
) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        label = f"{where}[{index}]"
        if not item.strip():
            rejections.append(
                MappingRejection("empty_finding", f"{label} is empty")
            )
            continue
        key = " ".join(item.casefold().split())
        if key in seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding", f"{label} duplicates an earlier entry"
                )
            )
        seen.add(key)
        check_coverage_language(item, label, rejections)
        _check_grounded_numbers(item, haystack_tokens, label, rejections)


# -- query proposal -----------------------------------------------------------


def check_queries(
    payload: Mapping[str, object], *, brief: ResearchBrief
) -> tuple[MappingRejection, ...]:
    rejections: list[MappingRejection] = []
    queries = payload["queries"]
    assert isinstance(queries, Sequence)  # schema-validated
    if not queries:
        return (
            MappingRejection(
                "empty_finding", "a query proposal must propose queries"
            ),
        )
    per_family: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(queries):
        assert isinstance(entry, Mapping)
        family = str(entry["family"])
        text = str(entry["text"]).strip()
        label = f"queries[{index}]"
        if not text or len(text) < 3:
            rejections.append(
                MappingRejection(
                    "empty_finding", f"{label} carries no usable query text"
                )
            )
            continue
        if len(text) > MAX_QUERY_TEXT_CHARS:
            rejections.append(
                MappingRejection(
                    "budget_violation",
                    f"{label} exceeds {MAX_QUERY_TEXT_CHARS} characters",
                )
            )
        key = (family, text.casefold())
        if key in seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{label} repeats an earlier query in the same family",
                )
            )
        seen.add(key)
        per_family[family] = per_family.get(family, 0) + 1
    for family, count in sorted(per_family.items()):
        if count > brief.max_queries_per_family:
            rejections.append(
                MappingRejection(
                    "budget_violation",
                    f"family {family} proposes {count} queries; the brief "
                    f"allows {brief.max_queries_per_family}",
                )
            )
    missing = sorted(
        family.value
        for family in REQUIRED_FAMILIES
        if family.value not in per_family
    )
    if missing:
        rejections.append(
            MappingRejection(
                "missing_family",
                f"required famil(ies) not covered: {', '.join(missing)}",
            )
        )
    return tuple(rejections)


# -- screening ----------------------------------------------------------------


def check_screening(
    payload: Mapping[str, object], *, expected_source_ids: Sequence[str]
) -> tuple[MappingRejection, ...]:
    """One batch: exactly one decision per expected source, no unknown
    ids, every decision reasoned."""
    rejections: list[MappingRejection] = []
    decisions = payload["decisions"]
    assert isinstance(decisions, Sequence)
    expected = set(expected_source_ids)
    seen: set[str] = set()
    for index, entry in enumerate(decisions):
        assert isinstance(entry, Mapping)
        source_id = str(entry["source_id"])
        label = f"decisions[{index}]"
        if source_id not in expected:
            rejections.append(
                MappingRejection(
                    "unknown_source",
                    f"{label} screens {source_id!r}, which is not in this "
                    f"batch",
                )
            )
            continue
        if source_id in seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{label} screens {source_id} a second time",
                )
            )
        seen.add(source_id)
        reason = str(entry["reason"])
        if not reason.strip():
            rejections.append(
                MappingRejection(
                    "empty_finding", f"{label} carries no reason"
                )
            )
        check_coverage_language(reason, f"{label}.reason", rejections)
    unscreened = sorted(expected - seen)
    if unscreened:
        rejections.append(
            MappingRejection(
                "missing_decision",
                f"no decision for source(s): {', '.join(unscreened)}",
            )
        )
    return tuple(rejections)


# -- extraction ---------------------------------------------------------------


def check_extraction(
    payload: Mapping[str, object], *, source: LiteratureSource
) -> tuple[MappingRejection, ...]:
    rejections: list[MappingRejection] = []
    if str(payload["source_id"]) != source.id:
        rejections.append(
            MappingRejection(
                "unknown_source",
                f"the extraction names {payload['source_id']!r}; this call "
                f"is about {source.id}",
            )
        )
    location = str(payload["support_location"])
    if location == SupportLocation.FULL_TEXT.value:
        rejections.append(
            MappingRejection(
                "access_level_mismatch",
                "full_text support is claimed, but Task 5A retrieved at "
                "most an abstract",
            )
        )
    elif (
        location == SupportLocation.ABSTRACT.value
        and source.access_level is not AccessLevel.ABSTRACT
    ):
        rejections.append(
            MappingRejection(
                "access_level_mismatch",
                f"abstract support is claimed, but the source's access "
                f"level is {source.access_level.value!r}",
            )
        )

    sufficient = bool(payload["sufficient_support"])
    reason = str(payload["insufficiency_reason"])
    claim_lists = (
        "methods",
        "metrics",
        "evaluation_protocols",
        "baselines",
        "reported_results",
        "future_work",
        "open_problems",
    )
    haystack = accessible_text_of(source)
    tokens = _number_tokens(haystack)
    if not sufficient:
        populated = [
            name
            for name in (*claim_lists, "datasets", "limitations")
            if _sequence(payload[name])
        ]
        if populated:
            rejections.append(
                MappingRejection(
                    "conflicting_record",
                    f"insufficient support is declared, yet "
                    f"{', '.join(populated)} carr(ies) claims",
                )
            )
        if not reason.strip():
            rejections.append(
                MappingRejection(
                    "empty_finding",
                    "insufficient support requires a reason",
                )
            )
        return tuple(rejections)

    if reason.strip():
        rejections.append(
            MappingRejection(
                "conflicting_record",
                "sufficient support and an insufficiency reason cannot "
                "both be present",
            )
        )
    if not any(
        _sequence(payload[name])
        for name in (*claim_lists, "datasets", "limitations")
    ):
        rejections.append(
            MappingRejection(
                "empty_finding",
                "sufficient support is declared, yet nothing was "
                "extracted; return the insufficiency outcome instead",
            )
        )
    for name in claim_lists:
        _check_texts(
            tuple(str(item) for item in _sequence(payload[name])),
            name,
            tokens,
            rejections,
        )
    _check_datasets(payload, haystack, tokens, rejections)
    _check_limitations(payload, tokens, rejections)
    return tuple(rejections)


def _check_datasets(
    payload: Mapping[str, object],
    haystack: str,
    tokens: frozenset[str],
    rejections: list[MappingRejection],
) -> None:
    seen: set[str] = set()
    for index, entry in enumerate(_sequence(payload["datasets"])):
        assert isinstance(entry, Mapping)
        label = f"datasets[{index}]"
        name = str(entry["name"]).strip()
        if not name:
            rejections.append(
                MappingRejection("empty_finding", f"{label} has no name")
            )
            continue
        if name.casefold() in seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding", f"{label} repeats {name!r}"
                )
            )
        seen.add(name.casefold())
        if name.casefold() not in haystack:
            rejections.append(
                MappingRejection(
                    "unsupported_claim",
                    f"{label} names {name!r}, which does not appear in the "
                    f"source's accessible text",
                )
            )
        for detail in ("version", "size", "url", "license"):
            value = str(entry[detail]).strip()
            if value and value.casefold() not in haystack:
                rejections.append(
                    MappingRejection(
                        "unsupported_claim",
                        f"{label}.{detail} reports {value!r}, which does "
                        f"not appear in the source's accessible text",
                    )
                )
        url = str(entry["url"]).strip()
        if url and not url.startswith(("http://", "https://")):
            rejections.append(
                MappingRejection(
                    "malformed_finding",
                    f"{label}.url {url!r} is not an http(s) URL",
                )
            )
        for field_name in ("task", "split", "subset", "preprocessing", "size"):
            _check_grounded_numbers(
                str(entry[field_name]),
                tokens,
                f"{label}.{field_name}",
                rejections,
            )
        check_coverage_language(
            str(entry["preprocessing"]), f"{label}.preprocessing", rejections
        )


def _check_limitations(
    payload: Mapping[str, object],
    tokens: frozenset[str],
    rejections: list[MappingRejection],
) -> None:
    texts = []
    for entry in _sequence(payload["limitations"]):
        assert isinstance(entry, Mapping)
        texts.append(str(entry["text"]))
    _check_texts(texts, "limitations", tokens, rejections)


# -- the field map ------------------------------------------------------------


def check_field_map(
    payload: Mapping[str, object],
    *,
    eras: Mapping[str, SourceEra],
    accessible: Mapping[str, str],
) -> tuple[MappingRejection, ...]:
    """``eras`` and ``accessible`` cover exactly the sufficiently
    extracted sources — the only universe the map may cite."""
    rejections: list[MappingRejection] = []
    themes = _sequence(payload["themes"])
    if not themes:
        rejections.append(
            MappingRejection(
                "empty_finding", "a field map requires at least one theme"
            )
        )
    theme_names: set[str] = set()
    for index, entry in enumerate(themes):
        assert isinstance(entry, Mapping)
        label = f"themes[{index}]"
        name = str(entry["name"]).strip()
        if name.casefold() in theme_names:
            rejections.append(
                MappingRejection(
                    "duplicate_finding", f"{label} repeats theme {name!r}"
                )
            )
        theme_names.add(name.casefold())
        cited = _check_group(entry, label, eras, accessible, rejections)
        if cited:
            expected = _expected_era(cited, eras)
            claimed = str(entry["era"])
            if claimed != expected.value:
                rejections.append(
                    MappingRejection(
                        "era_mismatch",
                        f"{label} claims era {claimed!r}, but its cited "
                        f"sources classify as {expected.value!r} under the "
                        f"brief's window",
                    )
                )
    for group_name in ("approaches", "evaluation_practices"):
        for index, entry in enumerate(_sequence(payload[group_name])):
            assert isinstance(entry, Mapping)
            _check_group(
                entry, f"{group_name}[{index}]", eras, accessible, rejections
            )
    declared = {
        str(entry["name"]).strip()
        for entry in themes
        if isinstance(entry, Mapping)
    }
    for index, entry in enumerate(_sequence(payload["relationships"])):
        assert isinstance(entry, Mapping)
        label = f"relationships[{index}]"
        from_theme = str(entry["from_theme"])
        to_theme = str(entry["to_theme"])
        for endpoint in (from_theme, to_theme):
            if endpoint not in declared:
                rejections.append(
                    MappingRejection(
                        "unknown_theme",
                        f"{label} references theme {endpoint!r}, which the "
                        f"map does not declare",
                    )
                )
        if from_theme == to_theme:
            rejections.append(
                MappingRejection(
                    "conflicting_record",
                    f"{label} relates theme {from_theme!r} to itself",
                )
            )
        note = str(entry["note"])
        if not note.strip():
            rejections.append(
                MappingRejection("empty_finding", f"{label} has no note")
            )
        check_coverage_language(note, f"{label}.note", rejections)
    return tuple(rejections)


def _check_group(
    entry: Mapping[str, object],
    label: str,
    eras: Mapping[str, SourceEra],
    accessible: Mapping[str, str],
    rejections: list[MappingRejection],
) -> tuple[str, ...]:
    """Shared checks for a named, summarized, source-grounded cluster.
    Returns the cited ids (empty on reference failure)."""
    name = str(entry["name"])
    summary = str(entry["summary"])
    if not name.strip():
        rejections.append(
            MappingRejection("empty_finding", f"{label} has no name")
        )
    if not summary.strip():
        rejections.append(
            MappingRejection("empty_finding", f"{label} has no summary")
        )
    check_coverage_language(name, f"{label}.name", rejections)
    check_coverage_language(summary, f"{label}.summary", rejections)
    cited = tuple(str(item) for item in _sequence(entry["source_ids"]))
    if not cited:
        rejections.append(
            MappingRejection(
                "missing_support",
                f"{label} cites no sources; synthesis must be grounded",
            )
        )
        return ()
    if len(cited) != len(set(cited)):
        rejections.append(
            MappingRejection(
                "duplicate_finding", f"{label} cites a source twice"
            )
        )
    unknown = sorted(set(cited) - set(eras))
    if unknown:
        rejections.append(
            MappingRejection(
                "unknown_source",
                f"{label} cites {', '.join(unknown)}, not among the "
                f"sufficiently extracted sources",
            )
        )
        return ()
    haystack_tokens = _number_tokens(
        "\n".join(accessible[source_id] for source_id in cited)
    )
    _check_grounded_numbers(
        summary, haystack_tokens, f"{label}.summary", rejections
    )
    return cited


def _expected_era(
    cited: Sequence[str], eras: Mapping[str, SourceEra]
) -> ThemeEra:
    kinds = {eras[source_id] for source_id in cited}
    if kinds == {SourceEra.RECENT}:
        return ThemeEra.RECENT
    if kinds == {SourceEra.FOUNDATIONAL}:
        return ThemeEra.FOUNDATIONAL
    return ThemeEra.BOTH


# -- the problem inventory ----------------------------------------------------


def check_inventory(
    payload: Mapping[str, object],
    *,
    eras: Mapping[str, SourceEra],
    accessible: Mapping[str, str],
) -> tuple[MappingRejection, ...]:
    rejections: list[MappingRejection] = []
    problems = _sequence(payload["problems"])
    if not problems:
        rejections.append(
            MappingRejection(
                "empty_finding",
                "an inventory requires at least one problem",
            )
        )
    statements: set[str] = set()
    for index, entry in enumerate(problems):
        assert isinstance(entry, Mapping)
        label = f"problems[{index}]"
        statement = str(entry["statement"])
        grounding = str(entry["grounding"])
        if not statement.strip():
            rejections.append(
                MappingRejection("empty_finding", f"{label} has no statement")
            )
            continue
        key = " ".join(statement.casefold().split())
        if key in statements:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{label} restates an earlier problem",
                )
            )
        statements.add(key)
        if not grounding.strip():
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{label} does not say how its sources ground it",
                )
            )
        supporting = tuple(
            str(item) for item in _sequence(entry["supporting_source_ids"])
        )
        conflicting = tuple(
            str(item) for item in _sequence(entry["conflicting_source_ids"])
        )
        if not supporting:
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{label} cites no supporting sources",
                )
            )
        unknown = sorted(
            (set(supporting) | set(conflicting)) - set(eras)
        )
        if unknown:
            rejections.append(
                MappingRejection(
                    "unknown_source",
                    f"{label} cites {', '.join(unknown)}, not among the "
                    f"sufficiently extracted sources",
                )
            )
            continue
        overlap = sorted(set(supporting) & set(conflicting))
        if overlap:
            rejections.append(
                MappingRejection(
                    "conflicting_record",
                    f"{label} lists {', '.join(overlap)} as both supporting "
                    f"and conflicting",
                )
            )
        if (
            str(entry["kind"]) == "conflicting_findings"
            and not conflicting
        ):
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{label} claims conflicting findings but names no "
                    f"conflicting source",
                )
            )
        haystack_tokens = _number_tokens(
            "\n".join(
                accessible[source_id]
                for source_id in (*supporting, *conflicting)
            )
        )
        for text, where in (
            (statement, f"{label}.statement"),
            (grounding, f"{label}.grounding"),
        ):
            check_coverage_language(text, where, rejections)
            _check_grounded_numbers(text, haystack_tokens, where, rejections)
    return tuple(rejections)


def _sequence(value: object) -> Sequence[object]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return value
