"""Deterministic gates over the selector's model output.

Two gates, one per model-authored payload: the comparative review and
the final decision. Both are pure functions from a parsed payload plus
trusted context to the complete list of rules that fired — every rule,
never first-fail — so a corrective call receives complete feedback. The
rejection vocabulary is shared with the mapping, ideation, and prior-art
gates deliberately: a rule that means the same thing keeps the same
name.

What the gates hold the model to is exactly what is checkable: ids must
name reviewed candidates, the verdict restatement must match the
recorded assessment, quotes must re-find verbatim in the rendered
records they claim to quote, numbers must appear in the rendered
context, and banned novelty and coverage language stays out. The ten
judgment fields are prose the gates never pretend to verify — judgment
is the model's seat here — they are only held to the text discipline.

The one thing no payload can express is a stop: the decision schema has
nowhere to put one, and a "selection" of a stamped-disqualified or
invented candidate is rejected here (``disqualified_selection``,
``unknown_candidate``) rather than recorded. Stops are computed by
trusted code from validated disqualifiers, never authored by the model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from ..ideation.gates import check_novelty_language
from ..mapping.gates import MappingRejection, check_coverage_language
from .records import (
    GROUND_DIMENSIONS,
    REVIEW_FIELDS,
    DisqualificationGround,
    DisqualifierDimension,
)

__all__ = [
    "MappingRejection",
    "check_comparative_review",
    "check_selection_decision",
]

# Mirrors of the mapping gates' private number helpers, kept private here
# for the same reason.
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")


def _number_tokens(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def _without_known_ids(text: str, known_ids: frozenset[str]) -> str:
    cleaned = text
    for identifier in sorted(known_ids, key=len, reverse=True):
        cleaned = cleaned.replace(identifier, " ")
    return cleaned


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _check_selection_text(
    text: str, where: str, rejections: list[MappingRejection]
) -> None:
    """The checks every model-authored selection text passes: non-empty,
    uncorrupted, no coverage claims, no novelty claims. Judgment prose
    is never verified beyond this — taste is not a rule."""
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
    cleaned = _without_known_ids(text, known_ids)
    for token in sorted(_number_tokens(cleaned)):
        if token not in haystack_tokens:
            rejections.append(
                MappingRejection(
                    "ungrounded_number",
                    f"{where} asserts the number {token!r}, which does "
                    f"not appear in the rendered candidates, assessments, "
                    f"constraints, or direction",
                )
            )


def _checked_text(
    text: str,
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
    where: str,
    rejections: list[MappingRejection],
) -> None:
    _check_selection_text(text, where, rejections)
    if text.strip():
        _check_grounded_numbers(
            text, haystack_tokens, known_ids, where, rejections
        )


def _entries(payload: Mapping[str, object], key: str) -> list[object]:
    # A validated payload arrives deep-frozen (arrays as tuples); a
    # hand-built one arrives as plain lists. Both are sequences.
    value = payload.get(key, ())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _text(entry: Mapping[str, object], key: str) -> str:
    value = entry.get(key, "")
    return value if isinstance(value, str) else str(value)


def check_comparative_review(
    payload: Mapping[str, object],
    *,
    candidate_blocks: Mapping[str, str],
    assessment_verdicts: Mapping[str, str],
    constraint_haystacks: Mapping[str, str],
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
) -> tuple[MappingRejection, ...]:
    """The stage-1 gate: every eligible candidate reviewed exactly once,
    every eligible pair compared exactly once, the recorded verdict
    restated exactly, every disqualifier attested at both ends, and the
    text discipline everywhere. ``candidate_blocks`` maps each eligible
    id to the exact rendered block the prompt carried — the attestation
    haystack is what the model saw, nothing else."""
    rejections: list[MappingRejection] = []
    reviewed: set[str] = set()
    for raw in _entries(payload, "reviews"):
        if not isinstance(raw, Mapping):
            rejections.append(
                MappingRejection(
                    "malformed_finding", "a review entry is not an object"
                )
            )
            continue
        candidate_id = _text(raw, "candidate_id")
        where = f"reviews[{candidate_id or '?'}]"
        if candidate_id not in candidate_blocks:
            rejections.append(
                MappingRejection(
                    "unknown_candidate",
                    f"{where} names a candidate outside the eligible set; "
                    f"the eligible set is stamped by trusted code and is "
                    f"not negotiable",
                )
            )
            continue
        if candidate_id in reviewed:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} reviews the same candidate twice",
                )
            )
            continue
        reviewed.add(candidate_id)
        stated = _text(raw, "prior_art_verdict")
        expected = assessment_verdicts[candidate_id]
        if stated != expected:
            rejections.append(
                MappingRejection(
                    "misstated_verdict",
                    f"{where} restates the recorded prior-art verdict as "
                    f"{stated!r}; the assessment on record says "
                    f"{expected!r} — restate the record exactly",
                )
            )
        for name in REVIEW_FIELDS:
            _checked_text(
                _text(raw, name),
                haystack_tokens,
                known_ids,
                f"{where}.{name}",
                rejections,
            )
        _check_disqualifiers(
            raw,
            candidate_id=candidate_id,
            candidate_block=candidate_blocks[candidate_id],
            constraint_haystacks=constraint_haystacks,
            haystack_tokens=haystack_tokens,
            known_ids=known_ids,
            rejections=rejections,
        )
    for candidate_id in sorted(set(candidate_blocks) - reviewed):
        rejections.append(
            MappingRejection(
                "missing_decision",
                f"eligible candidate {candidate_id} was never reviewed; "
                f"every eligible candidate gets exactly one review",
            )
        )
    _check_pairs(
        payload,
        eligible_ids=frozenset(candidate_blocks),
        haystack_tokens=haystack_tokens,
        known_ids=known_ids,
        rejections=rejections,
    )
    return tuple(rejections)


def _check_disqualifiers(
    review: Mapping[str, object],
    *,
    candidate_id: str,
    candidate_block: str,
    constraint_haystacks: Mapping[str, str],
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
    rejections: list[MappingRejection],
) -> None:
    grounds_seen: set[DisqualificationGround] = set()
    for raw in _entries(review, "disqualifiers"):
        where = f"reviews[{candidate_id}].disqualifiers"
        if not isinstance(raw, Mapping):
            rejections.append(
                MappingRejection(
                    "malformed_finding",
                    f"{where} holds an entry that is not an object",
                )
            )
            continue
        try:
            ground = DisqualificationGround(_text(raw, "ground"))
            dimension = DisqualifierDimension(_text(raw, "dimension"))
        except ValueError:
            rejections.append(
                MappingRejection(
                    "malformed_finding",
                    f"{where} names an unknown ground or dimension",
                )
            )
            continue
        where = f"{where}[{ground.value}]"
        if ground in grounds_seen:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} disqualifies on the same ground twice",
                )
            )
            continue
        grounds_seen.add(ground)
        if dimension not in GROUND_DIMENSIONS[ground]:
            rejections.append(
                MappingRejection(
                    "disqualifier_contradiction",
                    f"{where} quotes the {dimension.value} constraint, "
                    f"which cannot carry a {ground.value} objection",
                )
            )
            continue
        candidate_text = _text(raw, "candidate_text")
        if not candidate_text.strip():
            rejections.append(
                MappingRejection(
                    "empty_finding", f"{where}.candidate_text is empty"
                )
            )
        elif _normalized(candidate_text) not in _normalized(candidate_block):
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{where}.candidate_text does not appear verbatim in "
                    f"the candidate's rendered record; quote the "
                    f"candidate's own words — a paraphrase cannot attest "
                    f"a disqualification",
                )
            )
        constraint_text = _text(raw, "constraint_text")
        haystack = constraint_haystacks.get(dimension.value, "")
        if not constraint_text.strip():
            rejections.append(
                MappingRejection(
                    "empty_finding", f"{where}.constraint_text is empty"
                )
            )
        elif _normalized(constraint_text) not in _normalized(haystack):
            rejections.append(
                MappingRejection(
                    "unsupported_claim",
                    f"{where}.constraint_text does not appear verbatim in "
                    f"the {dimension.value} constraint on record; quote "
                    f"the constraint exactly",
                )
            )
        _checked_text(
            _text(raw, "why_unrepairable"),
            haystack_tokens,
            known_ids,
            f"{where}.why_unrepairable",
            rejections,
        )


def _check_pairs(
    payload: Mapping[str, object],
    *,
    eligible_ids: frozenset[str],
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
    rejections: list[MappingRejection],
) -> None:
    entries = _entries(payload, "pairwise_comparisons")
    expected = {
        (first, second)
        for index, first in enumerate(sorted(eligible_ids))
        for second in sorted(eligible_ids)[index + 1 :]
    }
    if len(entries) > len(expected):
        rejections.append(
            MappingRejection(
                "budget_violation",
                f"{len(entries)} pairwise comparisons submitted for "
                f"{len(expected)} eligible pairs; every pair is compared "
                f"exactly once",
            )
        )
    compared: set[tuple[str, str]] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            rejections.append(
                MappingRejection(
                    "malformed_finding",
                    "a pairwise comparison is not an object",
                )
            )
            continue
        first = _text(raw, "first_candidate_id")
        second = _text(raw, "second_candidate_id")
        where = f"pairwise_comparisons[{first or '?'}/{second or '?'}]"
        if first == second:
            rejections.append(
                MappingRejection(
                    "circular_finding",
                    f"{where} compares a candidate to itself",
                )
            )
            continue
        if first not in eligible_ids or second not in eligible_ids:
            rejections.append(
                MappingRejection(
                    "unknown_candidate",
                    f"{where} names a candidate outside the eligible set",
                )
            )
            continue
        key = (min(first, second), max(first, second))
        if key in compared:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} compares the same pair twice",
                )
            )
            continue
        compared.add(key)
        _checked_text(
            _text(raw, "comparison"),
            haystack_tokens,
            known_ids,
            f"{where}.comparison",
            rejections,
        )
    for first, second in sorted(expected - compared):
        rejections.append(
            MappingRejection(
                "missing_decision",
                f"pair {first} / {second} was never compared; every "
                f"eligible pair gets exactly one explicit comparison",
            )
        )


def check_selection_decision(
    payload: Mapping[str, object],
    *,
    eligible_ids: frozenset[str],
    disqualified_ids: frozenset[str],
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
) -> tuple[MappingRejection, ...]:
    """The stage-2 gate: exactly one undisqualified eligible winner,
    argued against every other contender, with the text discipline
    everywhere. A stamped-disqualified or invented id has no route to a
    record, and the schema offers no stop to express — a run stops only
    when trusted code concludes it from validated disqualifiers."""
    rejections: list[MappingRejection] = []
    selected = _text(payload, "selected_candidate_id")
    if selected not in eligible_ids:
        rejections.append(
            MappingRejection(
                "unknown_candidate",
                f"selected_candidate_id {selected!r} is not an eligible "
                f"candidate; the eligible set is stamped by trusted code",
            )
        )
    elif selected in disqualified_ids:
        rejections.append(
            MappingRejection(
                "disqualified_selection",
                f"selected_candidate_id {selected!r} carries a validated "
                f"disqualifier; a settled candidate cannot be chosen",
            )
        )
    argued: set[str] = set()
    for raw in _entries(payload, "why_selected_over"):
        if not isinstance(raw, Mapping):
            rejections.append(
                MappingRejection(
                    "malformed_finding",
                    "a why_selected_over entry is not an object",
                )
            )
            continue
        candidate_id = _text(raw, "candidate_id")
        where = f"why_selected_over[{candidate_id or '?'}]"
        if candidate_id not in eligible_ids:
            rejections.append(
                MappingRejection(
                    "unknown_candidate",
                    f"{where} names a candidate outside the eligible set",
                )
            )
            continue
        if candidate_id in disqualified_ids:
            rejections.append(
                MappingRejection(
                    "disqualified_selection",
                    f"{where} argues against a candidate a validated "
                    f"disqualifier already settled",
                )
            )
            continue
        if candidate_id == selected:
            rejections.append(
                MappingRejection(
                    "circular_finding",
                    f"{where} argues the winner against itself",
                )
            )
            continue
        if candidate_id in argued:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} argues the same alternative twice",
                )
            )
            continue
        argued.add(candidate_id)
        _checked_text(
            _text(raw, "reason"),
            haystack_tokens,
            known_ids,
            f"{where}.reason",
            rejections,
        )
    contenders = eligible_ids - disqualified_ids - {selected}
    for candidate_id in sorted(contenders - argued):
        rejections.append(
            MappingRejection(
                "missing_decision",
                f"contender {candidate_id} was never argued against; the "
                f"winner explains its edge over every alternative",
            )
        )
    for name in ("decisive_tradeoff", "first_experimental_objective"):
        _checked_text(
            _text(payload, name),
            haystack_tokens,
            known_ids,
            name,
            rejections,
        )
    for name in ("required_capabilities", "residual_risks"):
        entries = _entries(payload, name)
        if not entries:
            rejections.append(
                MappingRejection("empty_finding", f"{name} is empty")
            )
            continue
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            text = entry if isinstance(entry, str) else str(entry)
            if text in seen:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{name}[{index}] repeats an earlier entry",
                    )
                )
                continue
            seen.add(text)
            _checked_text(
                text,
                haystack_tokens,
                known_ids,
                f"{name}[{index}]",
                rejections,
            )
    return tuple(rejections)
