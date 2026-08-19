"""The deterministic candidate gates: every rule a model output must pass
before anything is recorded.

Two gates, one per model-authored payload — the extracted CFP direction
and the candidate portfolio — each pure: they read the payload, the
trusted snapshot and mapping records, and nothing else, and they return
*every* rule that fired so a corrective call receives complete feedback.
The rejection vocabulary is shared with the mapping gates deliberately:
ideation is mapping's consumer, and the corrective-call plumbing is
identical.

Three decisions carry the epistemic weight:

* **Grounding is scoped by claim kind.** A candidate's ``grounding``
  narrative describes what the cited records report, so every number in
  it must appear in those records' claim texts (or the addressed
  problems' own gated statements); its ``cfp_alignment`` describes the
  call, so its numbers must appear in the extracted direction. Numbers in
  predictions, falsifiers, and resource estimates are *design targets* —
  proposing new numbers is their job — and are exempt. The grounding
  surface is always the mapping run's own gated claim texts
  (:func:`claim_text_of`), never raw literature: what this package cannot
  read, it cannot pretend to have read. A known identifier is a name,
  not a number: exact ``lit_``/``prob_``/``thm_`` handles from the
  accepted context are stripped before number extraction, while the
  digits of an unknown or fabricated id still read as ungrounded
  numbers.

* **Novelty language is banned outright.** Every candidate's novelty
  status is structurally ``UNASSESSED``; text claiming an idea is novel,
  the first, unexplored, or state of the art asserts what a bounded run
  cannot know and Task 5D has not yet challenged.

* **Legitimate Unicode is preserved.** Names, technical terms, and
  dataset titles keep their real spelling; only control characters —
  the observed transport corruption — are rejected, via the mapping
  layer's integrity check. Nothing is transliterated.

Scientific taste is not a rule anywhere here: a thin portfolio, a dull
candidate, or an honest refusal with a grounded justification passes like
any other output — and has no route to a second call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from ..mapping.gates import (
    MappingRejection,
    check_coverage_language,
    check_text_integrity,
)
from ..mapping.records import ExtractionRecord, ProblemEntry
from .direction import CfpSnapshot

#: Word-bounded phrases no candidate text may contain: they claim novelty
#: standing a generation run does not have. A bare "first" ("the first
#: layer", "as a first step") is deliberately not banned — only its
#: claim-shaped uses are.
NOVELTY_PHRASES: Final = (
    "novel",
    "novelty",
    "unexplored",
    "understudied",
    "state of the art",
    "state-of-the-art",
    "sota",
    "no prior work",
    "first ever",
    "the first to",
    "first work",
    "first study",
    "for the first time",
)

_NOVELTY_PATTERN: Final = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in NOVELTY_PHRASES) + r")\b"
)

# Mirrors of the mapping gates' private number helpers, kept private here
# for the same reason.
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")


def _number_tokens(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def claim_text_of(extraction: ExtractionRecord) -> str:
    """Everything one grounded extraction durably claims, as one
    case-folded haystack. This — and never the raw literature source,
    which this package cannot read — is what a candidate's grounding may
    trace to."""
    parts = [
        *extraction.methods,
        *(
            text
            for dataset in extraction.datasets
            for text in (
                dataset.name,
                dataset.task,
                dataset.version,
                dataset.split,
                dataset.subset,
                dataset.preprocessing,
                dataset.size,
                dataset.url,
                dataset.license,
            )
        ),
        *extraction.metrics,
        *extraction.evaluation_protocols,
        *extraction.baselines,
        *extraction.reported_results,
        *(entry.text for entry in extraction.limitations),
        *extraction.future_work,
        *extraction.open_problems,
    ]
    return "\n".join(part for part in parts if part).casefold()


def check_novelty_language(
    text: str, where: str, rejections: list[MappingRejection]
) -> None:
    """Reject novelty claims wherever they appear: novelty stays
    ``UNASSESSED`` until the prior-art challenge, so asserting it is a
    gate violation, not a style issue."""
    for phrase in sorted(set(_NOVELTY_PATTERN.findall(text.casefold()))):
        rejections.append(
            MappingRejection(
                "novelty_claim",
                f"{where} claims novelty standing ({phrase!r}) that stays "
                f"unassessed until the prior-art challenge; say what the "
                f"mapped records do not report instead: {text!r}",
            )
        )


def _check_candidate_text(
    text: str, where: str, rejections: list[MappingRejection]
) -> None:
    """The checks every model-authored candidate text passes: non-empty,
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
    where: str,
    rejections: list[MappingRejection],
) -> None:
    for token in sorted(_number_tokens(text)):
        if token not in haystack_tokens:
            rejections.append(
                MappingRejection(
                    "ungrounded_number",
                    f"{where} asserts the number {token!r}, which does not "
                    f"appear in the cited records' claim texts",
                )
            )


def _without_known_ids(text: str, known_ids: frozenset[str]) -> str:
    """Strip exact occurrences of known durable identifiers before
    number extraction: an id is a name, not a numerical claim, yet its
    hexadecimal tail reads as digit runs. Observed live (Task 5C,
    2026-08-19): ``lit_`` source ids pasted into grounding prose fired
    ``ungrounded_number`` 51 times on fragments of their own hex. Only
    identifiers the trusted context actually knows are stripped — the
    digits of a fabricated or unknown id still surface as ungrounded
    numbers, which is the honest fail-closed outcome — and stripping
    replaces the id with a space, so a real number standing beside a
    known id is still validated."""
    cleaned = text
    for identifier in sorted(known_ids, key=len, reverse=True):
        cleaned = cleaned.replace(identifier, " ")
    return cleaned


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def check_direction(
    payload: Mapping[str, object], *, snapshot: CfpSnapshot
) -> tuple[MappingRejection, ...]:
    """The direction gate: the scope is extractor synthesis and gets the
    full text discipline (coverage, novelty, integrity) plus numbers
    grounded in the snapshot; topics, constraints, and dates are the
    call's own words — quoted, so exempt from the claim-language rules —
    and must appear verbatim (whitespace-normalized, case-insensitive)
    in the snapshot text: extraction, not invention."""
    rejections: list[MappingRejection] = []
    haystack = _normalized(snapshot.text)
    tokens = _number_tokens(snapshot.text)
    scope = str(payload["scope"])
    _check_candidate_text(scope, "scope", rejections)
    _check_grounded_numbers(scope, tokens, "scope", rejections)
    topics = tuple(str(item) for item in _sequence(payload["topics"]))
    if not topics:
        rejections.append(
            MappingRejection(
                "empty_finding", "a direction must name at least one topic"
            )
        )
    for name, items in (
        ("topics", topics),
        (
            "constraints",
            tuple(str(item) for item in _sequence(payload["constraints"])),
        ),
        (
            "relevant_dates",
            tuple(
                str(item) for item in _sequence(payload["relevant_dates"])
            ),
        ),
    ):
        seen: set[str] = set()
        for index, item in enumerate(items):
            where = f"{name}[{index}]"
            if not item.strip():
                rejections.append(
                    MappingRejection("empty_finding", f"{where} is empty")
                )
                continue
            check_text_integrity(item, where, rejections)
            key = _normalized(item)
            if key in seen:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{where} duplicates an earlier entry",
                    )
                )
            seen.add(key)
            if key not in haystack:
                rejections.append(
                    MappingRejection(
                        "unsupported_claim",
                        f"{where} reports {item!r}, which does not appear "
                        f"verbatim in the supplied call text; copy the "
                        f"call's exact wording",
                    )
                )
    return tuple(rejections)


def check_candidates(
    payload: Mapping[str, object],
    *,
    problems: Mapping[str, ProblemEntry],
    themes: Mapping[str, str],
    direction_topics: Sequence[str],
    direction_text: str,
    accessible: Mapping[str, str],
    max_candidates: int,
) -> tuple[MappingRejection, ...]:
    """The portfolio gate. ``problems`` maps each derived problem key to
    its inventory entry, ``themes`` each derived theme key to its exact
    name, and ``accessible`` each grounded source id to its claim text —
    together the only universe a candidate may cite. ``direction_topics``
    and ``direction_text`` come from the accepted direction record, the
    only universe ``aligned_topics`` and ``cfp_alignment`` may draw on."""
    rejections: list[MappingRejection] = []
    candidates = _sequence(payload["candidates"])
    refusal = str(payload["refusal_justification"])
    rationale = str(payload["diversity_rationale"])
    known_ids = (
        frozenset(problems) | frozenset(themes) | frozenset(accessible)
    )

    if not candidates:
        if not refusal.strip():
            return (
                MappingRejection(
                    "empty_finding",
                    "zero candidates require a grounded refusal "
                    "justification; an unexplained empty portfolio is not "
                    "an outcome",
                ),
            )
        _check_candidate_text(
            refusal, "refusal_justification", rejections
        )
        _check_grounded_numbers(
            _without_known_ids(refusal, known_ids),
            _number_tokens(
                "\n".join(
                    (
                        *accessible.values(),
                        *(
                            f"{p.statement}\n{p.grounding}".casefold()
                            for p in problems.values()
                        ),
                    )
                )
            ),
            "refusal_justification",
            rejections,
        )
        if rationale.strip():
            rejections.append(
                MappingRejection(
                    "conflicting_record",
                    "a refusal carries no diversity rationale",
                )
            )
        return tuple(rejections)

    if refusal.strip():
        rejections.append(
            MappingRejection(
                "conflicting_record",
                "a candidate portfolio and a refusal justification cannot "
                "both be present",
            )
        )
    if len(candidates) > max_candidates:
        rejections.append(
            MappingRejection(
                "budget_violation",
                f"the portfolio proposes {len(candidates)} candidates; the "
                f"directive allows {max_candidates}",
            )
        )
    _check_candidate_text(rationale, "diversity_rationale", rejections)

    seen_titles: set[str] = set()
    seen_questions: set[str] = set()
    seen_hypotheses: set[str] = set()
    seen_shapes: set[tuple[frozenset[str], str]] = set()
    for index, entry in enumerate(candidates):
        assert isinstance(entry, Mapping)
        label = f"candidates[{index}]"
        refs_ok = _check_references(
            entry,
            label,
            problems=problems,
            themes=themes,
            direction_topics=direction_topics,
            accessible=accessible,
            rejections=rejections,
        )
        _check_texts_of(entry, label, rejections)
        _check_predictions(entry, label, rejections)
        for text, seen, what in (
            (str(entry["title"]), seen_titles, "title"),
            (
                str(entry["research_question"]),
                seen_questions,
                "research question",
            ),
            (str(entry["hypothesis"]), seen_hypotheses, "hypothesis"),
        ):
            key = _normalized(text)
            if key and key in seen:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{label} repeats an earlier candidate's {what}",
                    )
                )
            seen.add(key)
        shape = (
            frozenset(str(k) for k in _sequence(entry["problem_keys"])),
            _normalized(str(entry["mechanism"])),
        )
        if shape in seen_shapes:
            rejections.append(
                MappingRejection(
                    "insufficient_diversity",
                    f"{label} targets the same problems with the same "
                    f"mechanism as an earlier candidate; candidates must "
                    f"differ substantively, not in wording",
                )
            )
        seen_shapes.add(shape)
        if not refs_ok:
            continue
        haystack = "\n".join(
            (
                *(
                    accessible[str(source_id)]
                    for source_id in _sequence(entry["cited_source_ids"])
                ),
                *(
                    (
                        f"{problems[str(key)].statement}\n"
                        f"{problems[str(key)].grounding}"
                    ).casefold()
                    for key in _sequence(entry["problem_keys"])
                ),
            )
        )
        tokens = _number_tokens(haystack)
        _check_grounded_numbers(
            _without_known_ids(str(entry["grounding"]), known_ids),
            tokens,
            f"{label}.grounding",
            rejections,
        )
        _check_grounded_numbers(
            _without_known_ids(str(entry["cfp_alignment"]), known_ids),
            _number_tokens(direction_text.casefold()),
            f"{label}.cfp_alignment",
            rejections,
        )
        _check_datasets(entry, label, haystack, rejections)
    return tuple(rejections)


def _check_references(
    entry: Mapping[str, object],
    label: str,
    *,
    problems: Mapping[str, ProblemEntry],
    themes: Mapping[str, str],
    direction_topics: Sequence[str],
    accessible: Mapping[str, str],
    rejections: list[MappingRejection],
) -> bool:
    """Every canonical handle the candidate cites, checked against the
    rendered universe. Returns False on any reference failure, so the
    haystack-dependent checks are skipped rather than crashed."""
    ok = True
    problem_keys = tuple(str(k) for k in _sequence(entry["problem_keys"]))
    theme_keys = tuple(str(k) for k in _sequence(entry["theme_keys"]))
    cited = tuple(str(s) for s in _sequence(entry["cited_source_ids"]))
    topics = tuple(str(t) for t in _sequence(entry["aligned_topics"]))
    for what, items, rule, universe, advice in (
        (
            "problem_keys",
            problem_keys,
            "unknown_problem",
            frozenset(problems),
            "repeat the exact prob_ key shown for a listed problem, never "
            "an index, a shorthand, or a restated sentence",
        ),
        (
            "theme_keys",
            theme_keys,
            "unknown_theme",
            frozenset(themes),
            "repeat the exact thm_ key shown for a listed theme, never a "
            "shorthand label",
        ),
        (
            "cited_source_ids",
            cited,
            "unknown_source",
            frozenset(accessible),
            "cite only the listed lit_ ids",
        ),
        (
            "aligned_topics",
            topics,
            "unknown_topic",
            frozenset(direction_topics),
            "repeat a topic string exactly as the extracted direction "
            "lists it",
        ),
    ):
        if not items:
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{label} lists no {what}; a candidate must be "
                    f"grounded",
                )
            )
            ok = False
            continue
        if len(set(items)) != len(items):
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{label}.{what} repeats an entry",
                )
            )
        unknown = sorted(set(items) - universe)
        if unknown:
            rejections.append(
                MappingRejection(
                    rule,
                    f"{label}.{what} references {', '.join(unknown)}, not "
                    f"in this run's records; {advice}",
                )
            )
            ok = False
    if not ok:
        return False
    for key in problem_keys:
        problem = problems[key]
        grounding_ids = set(problem.supporting_source_ids) | set(
            problem.conflicting_source_ids
        )
        if not grounding_ids & set(cited):
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{label} addresses problem {key} but cites none of "
                    f"the sources that ground it",
                )
            )
    return True


def _check_texts_of(
    entry: Mapping[str, object],
    label: str,
    rejections: list[MappingRejection],
) -> None:
    for name in (
        "title",
        "research_question",
        "proposed_contribution",
        "mechanism",
        "hypothesis",
        "grounding",
        "evaluation_protocol",
        "cfp_alignment",
        "uncertainty",
    ):
        _check_candidate_text(
            str(entry[name]), f"{label}.{name}", rejections
        )
    for name in ("metrics", "baselines", "ablations", "risks", "search_terms"):
        items = tuple(str(item) for item in _sequence(entry[name]))
        if not items:
            rejections.append(
                MappingRejection(
                    "empty_finding", f"{label}.{name} lists nothing"
                )
            )
        seen: set[str] = set()
        for index, item in enumerate(items):
            where = f"{label}.{name}[{index}]"
            _check_candidate_text(item, where, rejections)
            key = _normalized(item)
            if key and key in seen:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{where} duplicates an earlier entry",
                    )
                )
            seen.add(key)
    resources = entry["resources"]
    assert isinstance(resources, Mapping)
    for name in ("compute", "data", "implementation"):
        _check_candidate_text(
            str(resources[name]), f"{label}.resources.{name}", rejections
        )


def _check_predictions(
    entry: Mapping[str, object],
    label: str,
    rejections: list[MappingRejection],
) -> None:
    predictions = _sequence(entry["predictions"])
    if not predictions:
        rejections.append(
            MappingRejection(
                "empty_finding",
                f"{label}.predictions lists nothing; a candidate needs at "
                f"least one measurable prediction",
            )
        )
    hypothesis = _normalized(str(entry["hypothesis"]))
    seen: set[str] = set()
    for index, item in enumerate(predictions):
        assert isinstance(item, Mapping)
        where = f"{label}.predictions[{index}]"
        text = str(item["text"])
        falsifier = str(item["falsifier"])
        _check_candidate_text(text, f"{where}.text", rejections)
        _check_candidate_text(falsifier, f"{where}.falsifier", rejections)
        if _normalized(text) and _normalized(text) == _normalized(falsifier):
            rejections.append(
                MappingRejection(
                    "circular_finding",
                    f"{where} states its falsifier as itself; a falsifier "
                    f"names the observation that would refute the "
                    f"prediction",
                )
            )
        for part, what in ((text, "prediction"), (falsifier, "falsifier")):
            if hypothesis and _normalized(part) == hypothesis:
                rejections.append(
                    MappingRejection(
                        "circular_finding",
                        f"{where} restates the hypothesis as its {what}; "
                        f"the hypothesis must be independently falsifiable",
                    )
                )
        key = _normalized(text)
        if key and key in seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} duplicates an earlier prediction",
                )
            )
        seen.add(key)


def _check_datasets(
    entry: Mapping[str, object],
    label: str,
    haystack: str,
    rejections: list[MappingRejection],
) -> None:
    datasets = _sequence(entry["datasets"])
    if not datasets:
        rejections.append(
            MappingRejection(
                "empty_finding",
                f"{label}.datasets lists nothing; a candidate names its "
                f"data or its data requirements",
            )
        )
    seen: set[str] = set()
    for index, item in enumerate(datasets):
        assert isinstance(item, Mapping)
        where = f"{label}.datasets[{index}]"
        name = str(item["name"]).strip()
        if not name:
            rejections.append(
                MappingRejection("empty_finding", f"{where} has no name")
            )
            continue
        check_text_integrity(name, f"{where}.name", rejections)
        if name.casefold() in seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding", f"{where} repeats {name!r}"
                )
            )
        seen.add(name.casefold())
        _check_candidate_text(
            str(item["role"]), f"{where}.role", rejections
        )
        if (
            str(item["status"]) == "existing"
            and name.casefold() not in haystack
        ):
            rejections.append(
                MappingRejection(
                    "unsupported_claim",
                    f"{where} presents {name!r} as existing, but no cited "
                    f"record reports it; mark it new_requirement or cite a "
                    f"record that reports it",
                )
            )


def _sequence(value: object) -> Sequence[object]:
    assert isinstance(value, Sequence) and not isinstance(value, str)
    return value
