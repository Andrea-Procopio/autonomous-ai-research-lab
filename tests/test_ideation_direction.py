"""The CFP ingress: immutable hashed snapshots, direction records, and
the deterministic extraction gate that holds every reported item to the
supplied text. No network, no model."""

from __future__ import annotations

import hashlib

import pytest

from autonomous_research_lab.ideation.direction import (
    CfpSnapshot,
    DirectionRecord,
)
from autonomous_research_lab.ideation.gates import check_direction
from autonomous_research_lab.mapping.records import CallProvenance

CALL_TEXT = (
    "Workshop on In-Context Learning.\n"
    "We invite submissions on the following topics:\n"
    "- mechanisms of in-context learning\n"
    "- efficient adaptation and fine-tuning\n"
    "Submissions are limited to 9 pages.\n"
    "Deadline: 30 September 2026.\n"
)

SNAPSHOT = CfpSnapshot(
    source_url="https://example.org/workshop/cfp",
    supplied_at="2026-08-19T10:00:00",
    text=CALL_TEXT,
)


def _provenance() -> CallProvenance:
    return CallProvenance(
        request_fingerprint="mreq_1",
        response_id="mcall_1",
        provider="fake",
        requested_model="model-x",
        served_model="model-x",
        provider_request_id=None,
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=200,
        repair_count=0,
    )


def _payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "scope": (
            "A workshop call about in-context learning, asking for work "
            "on mechanisms and on efficient adaptation."
        ),
        "topics": [
            "mechanisms of in-context learning",
            "efficient adaptation and fine-tuning",
        ],
        "constraints": ["Submissions are limited to 9 pages."],
        "relevant_dates": ["30 September 2026"],
    }
    values.update(overrides)
    return values


def _rules(rejections: tuple[object, ...]) -> set[str]:
    return {r.rule for r in rejections}  # type: ignore[attr-defined]


# -- the snapshot -------------------------------------------------------------


def test_a_snapshot_is_immutable_timestamped_and_hashed() -> None:
    assert SNAPSHOT.id.startswith("cfp_")
    expected = hashlib.sha256(CALL_TEXT.encode("utf-8")).hexdigest()
    assert SNAPSHOT.text_sha256 == expected
    again = CfpSnapshot(
        source_url=SNAPSHOT.source_url,
        supplied_at=SNAPSHOT.supplied_at,
        text=CALL_TEXT,
        text_sha256=expected,
    )
    assert again == SNAPSHOT
    later = CfpSnapshot(
        source_url=SNAPSHOT.source_url,
        supplied_at="2026-08-19T11:00:00",
        text=CALL_TEXT,
    )
    # Two supplies of one text are two snapshots: provenance is part of
    # the identity.
    assert later.id != SNAPSHOT.id


def test_a_snapshot_whose_hash_disagrees_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="hash does not match"):
        CfpSnapshot(
            source_url="https://example.org/cfp",
            supplied_at="2026-08-19T10:00:00",
            text=CALL_TEXT,
            text_sha256="0" * 64,
        )


def test_a_malformed_snapshot_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="where its text came from"):
        CfpSnapshot(
            source_url=" ", supplied_at="2026-08-19T10:00:00", text="x"
        )
    with pytest.raises(ValueError, match="ISO timestamp"):
        CfpSnapshot(
            source_url="https://example.org",
            supplied_at="yesterday",
            text="x",
        )
    with pytest.raises(ValueError, match="supplied text"):
        CfpSnapshot(
            source_url="https://example.org",
            supplied_at="2026-08-19T10:00:00",
            text="   ",
        )
    with pytest.raises(ValueError, match="at most"):
        CfpSnapshot(
            source_url="https://example.org",
            supplied_at="2026-08-19T10:00:00",
            text="x" * 200_001,
        )


# -- the direction record -----------------------------------------------------


def test_a_direction_requires_scope_and_topics() -> None:
    with pytest.raises(ValueError, match="scope"):
        DirectionRecord(
            run_id="idg_1",
            snapshot_id=SNAPSHOT.id,
            scope="  ",
            topics=("a",),
            constraints=(),
            relevant_dates=(),
            provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="topic"):
        DirectionRecord(
            run_id="idg_1",
            snapshot_id=SNAPSHOT.id,
            scope="About in-context learning.",
            topics=(),
            constraints=(),
            relevant_dates=(),
            provenance=_provenance(),
        )


def test_direction_identity_is_deterministic() -> None:
    def build(scope: str) -> DirectionRecord:
        return DirectionRecord(
            run_id="idg_1",
            snapshot_id=SNAPSHOT.id,
            scope=scope,
            topics=("mechanisms of in-context learning",),
            constraints=(),
            relevant_dates=(),
            provenance=_provenance(),
        )

    first = build("About in-context learning.")
    assert first.id.startswith("dir_")
    assert first.id == build("About in-context learning.").id
    assert first.id != build("About adaptation.").id
    assert "mechanisms of in-context learning" in first.rendered_text()


# -- the extraction gate ------------------------------------------------------


def test_a_verbatim_direction_extraction_passes() -> None:
    assert check_direction(_payload(), snapshot=SNAPSHOT) == ()


def test_paraphrased_extraction_is_rejected() -> None:
    paraphrased = _payload(topics=["how in-context learning works"])
    assert _rules(check_direction(paraphrased, snapshot=SNAPSHOT)) == {
        "unsupported_claim"
    }


def test_scope_numbers_must_come_from_the_call() -> None:
    invented = _payload(
        scope="A workshop call expecting 200 submissions on mechanisms."
    )
    assert _rules(check_direction(invented, snapshot=SNAPSHOT)) == {
        "ungrounded_number"
    }
    grounded = _payload(
        scope="A workshop call limiting submissions to 9 pages."
    )
    assert check_direction(grounded, snapshot=SNAPSHOT) == ()


def test_corrupted_or_overclaiming_scope_is_rejected() -> None:
    corrupted = _payload(scope="mechanisms\x02adaptation")
    assert "corrupted_text" in _rules(
        check_direction(corrupted, snapshot=SNAPSHOT)
    )
    overclaiming = _payload(
        scope="An exhaustive review of in-context learning."
    )
    assert "coverage_language" in _rules(
        check_direction(overclaiming, snapshot=SNAPSHOT)
    )


def test_duplicate_and_empty_entries_are_rejected() -> None:
    duplicated = _payload(
        topics=[
            "mechanisms of in-context learning",
            "Mechanisms of  in-context learning",
        ]
    )
    assert "duplicate_finding" in _rules(
        check_direction(duplicated, snapshot=SNAPSHOT)
    )
    empty = _payload(topics=[])
    assert "empty_finding" in _rules(
        check_direction(empty, snapshot=SNAPSHOT)
    )
    blank = _payload(constraints=["   "])
    assert "empty_finding" in _rules(
        check_direction(blank, snapshot=SNAPSHOT)
    )
