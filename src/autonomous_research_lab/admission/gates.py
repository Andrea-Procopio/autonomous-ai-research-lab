"""The deterministic gate over the admitter's one model payload.

A pure function from the parsed operationalization payload plus trusted
context to the complete list of rules that fired — every rule, never
first-fail — so a corrective call receives complete feedback. The
rejection vocabulary is shared with the mapping, ideation, and selection
gates deliberately: a rule that means the same thing keeps the same
name.

What the gate holds the model to is exactly what is checkable: every
recorded prediction covered, nothing invented. The prediction being
encoded must equal a recorded prediction verbatim; the base metric must
be one the candidate declared; both arms must re-find in the rendered
candidate; the contrary observation must re-find in that prediction's
own falsifier and nowhere else; every support quote must re-find at its
named field path; numbers must appear in the rendered context; and any
id-shaped token must name a shown record — an id contains no decimal
digits, so the number gate alone would never see a fabricated
reference. ``condition`` is the one prose seat of model judgment; it is
held to the text discipline and its own support, never to taste.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from ..ideation.gates import check_novelty_language
from ..mapping.gates import MappingRejection, check_coverage_language
from .records import MAX_ARM_CHARS, MAX_ENCODINGS_PER_PREDICTION

__all__ = [
    "MappingRejection",
    "check_operationalization",
]

# Mirrors of the mapping gates' private number helpers, kept private here
# for the same reason.
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")

#: The house id shape (core/ids.py): a fabricated reference carries no
#: decimal digit, so it must be caught by shape, not by the number gate.
_REFERENCE: Final = re.compile(r"\b[a-z]+_[0-9a-f]{16}\b")

#: Success-guarantee language, word-bounded like the novelty pattern —
#: a plain substring would false-fire inside honest "uncertainty" prose.
_GUARANTEE: Final = re.compile(
    r"\b(?:guaranteed?|cannot fail|certain to succeed|will succeed)\b"
)

_PROSE_FIELDS: Final = (
    "prediction_text",
    "condition",
    "base_metric",
    "expected_higher_arm",
    "expected_lower_arm",
    "contrary_observation",
)


def _number_tokens(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def _without_known_ids(text: str, known_ids: frozenset[str]) -> str:
    cleaned = text
    for identifier in sorted(known_ids, key=len, reverse=True):
        cleaned = cleaned.replace(identifier, " ")
    return cleaned


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


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


def _check_admission_text(
    text: str, where: str, rejections: list[MappingRejection]
) -> None:
    """The checks every model-authored admission text passes: non-empty,
    no coverage claims, no novelty claims, no success guarantees.
    Judgment prose is never verified beyond this — taste is not a
    rule."""
    if not text.strip():
        rejections.append(
            MappingRejection("empty_finding", f"{where} is empty")
        )
        return
    check_coverage_language(text, where, rejections)
    check_novelty_language(text, where, rejections)
    if _GUARANTEE.search(text.casefold()):
        rejections.append(
            MappingRejection(
                "novelty_claim",
                f"{where} guarantees success; an admission states what "
                f"would be observed, never that it will succeed",
            )
        )


def _check_references(
    text: str,
    known_ids: frozenset[str],
    where: str,
    rejections: list[MappingRejection],
) -> None:
    for token in sorted(set(_REFERENCE.findall(text))):
        if token not in known_ids:
            rejections.append(
                MappingRejection(
                    "fabricated_reference",
                    f"{where} references {token}, which names no shown "
                    f"record",
                )
            )


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
                    f"not appear in the rendered candidate, direction, or "
                    f"directive statements",
                )
            )


def _checked_text(
    text: str,
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
    where: str,
    rejections: list[MappingRejection],
) -> None:
    _check_admission_text(text, where, rejections)
    if text.strip():
        _check_references(text, known_ids, where, rejections)
        _check_grounded_numbers(
            text, haystack_tokens, known_ids, where, rejections
        )


def check_operationalization(
    payload: Mapping[str, object],
    *,
    field_texts: Mapping[str, str],
    candidate_block: str,
    falsifier_by_prediction: Mapping[str, str],
    metrics: tuple[str, ...],
    haystack_tokens: frozenset[str],
    known_ids: frozenset[str],
) -> tuple[MappingRejection, ...]:
    """The one gate: every recorded prediction operationalized at least
    once and at most :data:`MAX_ENCODINGS_PER_PREDICTION` times, every
    quote re-found where it claims to live, no invented metric, arm,
    number, or reference. ``field_texts`` maps each groundable field
    path to the exact text the prompt carried (direction paths under a
    ``direction.`` prefix); ``falsifier_by_prediction`` maps each
    recorded prediction's normalized text to its falsifier."""
    rejections: list[MappingRejection] = []
    entries = _entries(payload, "operational_predictions")
    covered: dict[str, int] = dict.fromkeys(falsifier_by_prediction, 0)
    mechanical: set[tuple[str, str, str, str]] = set()
    normalized_block = _normalized(candidate_block)

    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            rejections.append(
                MappingRejection(
                    "malformed_finding",
                    f"operational_predictions[{index}] is not an object",
                )
            )
            continue
        where = f"operational_predictions[{index}]"
        for name in _PROSE_FIELDS:
            _checked_text(
                _text(raw, name),
                haystack_tokens,
                known_ids,
                f"{where}.{name}",
                rejections,
            )

        prediction_text = _text(raw, "prediction_text")
        matched = _normalized(prediction_text)
        if prediction_text.strip() and matched not in falsifier_by_prediction:
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{where}.prediction_text does not restate any "
                    f"recorded prediction verbatim; an encoding of a "
                    f"rewritten prediction encodes nothing on record",
                )
            )
            matched = ""
        if matched:
            covered[matched] += 1

        base_metric = _text(raw, "base_metric")
        if base_metric.strip() and base_metric not in metrics:
            rejections.append(
                MappingRejection(
                    "metric_not_declared",
                    f"{where}.base_metric {base_metric!r} is not among "
                    f"the candidate's declared metrics",
                )
            )

        higher = _text(raw, "expected_higher_arm")
        lower = _text(raw, "expected_lower_arm")
        for arm_name, arm in (
            ("expected_higher_arm", higher),
            ("expected_lower_arm", lower),
        ):
            if not arm.strip():
                continue
            if len(arm) > MAX_ARM_CHARS:
                rejections.append(
                    MappingRejection(
                        "malformed_finding",
                        f"{where}.{arm_name} exceeds {MAX_ARM_CHARS} "
                        f"characters; arms are embedded in the metric "
                        f"contract and stay short",
                    )
                )
            if _normalized(arm) not in normalized_block:
                rejections.append(
                    MappingRejection(
                        "missing_support",
                        f"{where}.{arm_name} is not the candidate's own "
                        f"wording; an arm the record does not contain "
                        f"rewrites the candidate",
                    )
                )
        if (
            higher.strip()
            and lower.strip()
            and _normalized(higher) == _normalized(lower)
        ):
            rejections.append(
                MappingRejection(
                    "circular_finding",
                    f"{where} compares an arm with itself; the "
                    f"difference commits to nothing",
                )
            )

        contrary = _text(raw, "contrary_observation")
        if matched and contrary.strip():
            falsifier = _normalized(falsifier_by_prediction[matched])
            if _normalized(contrary) not in falsifier:
                rejections.append(
                    MappingRejection(
                        "missing_support",
                        f"{where}.contrary_observation is not re-found "
                        f"in that prediction's recorded falsifier; the "
                        f"contrary outcome is the record's, not the "
                        f"model's",
                    )
                )

        key = (_text(raw, "condition"), base_metric, higher, lower)
        if key in mechanical:
            rejections.append(
                MappingRejection(
                    "duplicate_finding",
                    f"{where} repeats another encoding's condition, "
                    f"metric, and arms; the derived predictions would "
                    f"silently share one identity",
                )
            )
        mechanical.add(key)

        support = _entries(raw, "support")
        if not support:
            rejections.append(
                MappingRejection(
                    "missing_support",
                    f"{where} carries no grounded support link",
                )
            )
        seen_links: set[tuple[str, str]] = set()
        for link_index, link in enumerate(support):
            if not isinstance(link, Mapping):
                rejections.append(
                    MappingRejection(
                        "malformed_finding",
                        f"{where}.support[{link_index}] is not an object",
                    )
                )
                continue
            link_where = f"{where}.support[{link_index}]"
            source = _text(link, "source")
            field_path = _text(link, "field_path")
            quote = _text(link, "quote")
            if (field_path, quote) in seen_links:
                rejections.append(
                    MappingRejection(
                        "duplicate_finding",
                        f"{link_where} repeats an earlier support link",
                    )
                )
            seen_links.add((field_path, quote))
            if field_path not in field_texts:
                rejections.append(
                    MappingRejection(
                        "unknown_field",
                        f"{link_where} names field path {field_path!r}, "
                        f"which is not a groundable field of the shown "
                        f"records",
                    )
                )
                continue
            expected_prefix = source == "direction"
            if field_path.startswith("direction.") is not expected_prefix:
                rejections.append(
                    MappingRejection(
                        "unknown_field",
                        f"{link_where} labels {field_path!r} as coming "
                        f"from the {source or '?'}; the path says "
                        f"otherwise",
                    )
                )
                continue
            if not quote.strip():
                rejections.append(
                    MappingRejection(
                        "empty_finding", f"{link_where}.quote is empty"
                    )
                )
                continue
            if _normalized(quote) not in _normalized(
                field_texts[field_path]
            ):
                rejections.append(
                    MappingRejection(
                        "unsupported_claim",
                        f"{link_where} quotes text not present at "
                        f"{field_path!r}; support is verbatim or it is "
                        f"not support",
                    )
                )

    for text, count in covered.items():
        if count == 0:
            rejections.append(
                MappingRejection(
                    "missing_decision",
                    f"the recorded prediction {text[:60]!r} was never "
                    f"operationalized; admission encodes the whole "
                    f"record or refuses",
                )
            )
        if count > MAX_ENCODINGS_PER_PREDICTION:
            rejections.append(
                MappingRejection(
                    "budget_violation",
                    f"the recorded prediction {text[:60]!r} carries "
                    f"{count} encodings; at most "
                    f"{MAX_ENCODINGS_PER_PREDICTION} are allowed",
                )
            )

    return tuple(rejections)
