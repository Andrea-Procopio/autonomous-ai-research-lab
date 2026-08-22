"""The faithfulness reviewer: criticism grounded or refused.

Unit half: the deterministic gates, grounding, verdict derivation, the
schema, the records, the store, and the reviewer against the scripted
fake provider. Composition half: the full revise cycle over a stub lab,
every crash window's recovery, and the bounded-cycle invariant. The
integration half walks the CI lab once (module-scoped) and reviews the
scripted draft end to end.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from autonomous_research_lab.control.cli import OK, REFUSED, main
from autonomous_research_lab.control.review import review_manuscript
from autonomous_research_lab.control.stage import StageName
from autonomous_research_lab.publication.manuscript import (
    PROSE_SECTIONS,
    AuthorCall,
    Manuscript,
    ProseSections,
)
from autonomous_research_lab.publication.packet import (
    AssessmentRecord,
    Bibliography,
    BibliographyEntry,
    ClaimFinding,
    EvidencePacket,
    EvidenceRow,
    Provenance,
    RegisteredScience,
    SpendSummary,
    VerifySummary,
)
from autonomous_research_lab.publication.review import (
    MODEL_ISSUES,
    ReviewError,
    ReviewFinding,
    ReviewRecord,
    ReviewRejectedError,
    ReviewVerdict,
    RevisionRecord,
    derive_verdict,
    deterministic_findings,
    ground_findings,
    packet_identifiers,
    review_schema,
)
from autonomous_research_lab.publication.reviewer import FaithfulnessReviewer
from autonomous_research_lab.publication.store import (
    AmbiguousHeadError,
    ManuscriptStore,
    ReviewIntegrityError,
    ReviewStore,
    head_for,
)
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    StructuredOutputError,
    UsageLedger,
)
from examples.vision_chain import walk
from examples.vision_lab import ci_lab

SOURCE_ID = "lits_53b0e6573b48e679"


def reportable_packet() -> EvidencePacket:
    """One claim, assessed SUPPORTED — licensed and unlicensed verdict
    words are both expressible against it."""
    return EvidencePacket(
        provenance=Provenance(
            investigation_id="inv_1",
            label="a label",
            config_id="cfg_1",
            brief_topic="a topic",
            run_id="run_1",
            run_record_id="rune_1",
            authority="an operator",
            head_state_id="st_1",
            spend=SpendSummary(
                granted_wall_clock_seconds=600.0,
                granted_gpu_hours=0.0,
                granted_usd=1.0,
                granted_model_tokens=10_000,
                remaining_wall_clock_seconds=500.0,
                remaining_gpu_hours=0.0,
                remaining_usd=1.0,
                remaining_model_tokens=9_000,
                stage_model_calls=3,
                stage_input_tokens=100,
                stage_output_tokens=50,
            ),
            verified=VerifySummary(
                states_checked=1,
                results_checked=0,
                evidence_checked=0,
                blobs_checked=0,
            ),
        ),
        science=RegisteredScience(
            admission_record_id="adm_1",
            question_id="q_1",
            question="does it hold?",
            hypothesis_id="hyp_1",
            hypothesis="it holds",
            mechanical_reading="sign_only",
        ),
        claims=(
            ClaimFinding(
                claim_id="clm_1",
                statement="it holds",
                scope="held out",
                figures_check="restated-from-record",
                assessment=AssessmentRecord(
                    assessment_id="asm_1",
                    verdict="supported",
                    method="statistician:exact-sign-v1",
                    rationale="a recorded rationale",
                    scope="held out",
                ),
                evidence_rows=(
                    EvidenceRow(
                        evidence_id="ev_1",
                        result_id="res_1",
                        seed=11,
                        relation="supports",
                        standing="verified_evidence",
                        metrics=(("accuracy gap", 0.85),),
                    ),
                ),
            ),
        ),
        bibliography=Bibliography(
            candidate_id="idea_1",
            candidate_title="Trained versus random",
            research_question="does it hold?",
            entries=(
                BibliographyEntry(
                    source_id=SOURCE_ID,
                    title="A cited paper",
                    authors=("Ada Lovelace",),
                    venue="A venue",
                    year=2016,
                    doi="",
                    arxiv_id="",
                    url="https://example.invalid/1",
                    access_level="abstract",
                ),
            ),
        ),
    )


def prose(**overrides: str) -> dict[str, str]:
    sections = {
        name: "The claim is supported on held out data."
        for name in PROSE_SECTIONS
    }
    sections.update(overrides)
    return sections


def manuscript_for(
    packet: EvidencePacket, **section_overrides: str
) -> Manuscript:
    return Manuscript(
        packet_id=packet.packet_id,
        sections=ProseSections(**prose(**section_overrides)),
        call=AuthorCall(
            request_fingerprint="mreq_1",
            response_id="resp_1",
            provider="fake",
            requested_model="fake-model-1",
            served_model="fake-model-1",
            provider_request_id=None,
            latency_seconds=0.25,
            input_tokens=100,
            output_tokens=50,
            repair_count=0,
        ),
    )


class TestDeterministicFindings:
    def test_a_faithful_draft_has_none(self) -> None:
        packet = reportable_packet()
        assert deterministic_findings(manuscript_for(packet), packet) == ()

    def test_forbidden_strength_fires_verbatim(self) -> None:
        packet = reportable_packet()
        found = deterministic_findings(
            manuscript_for(
                packet, discussion="the gain is Statistically  Significant"
            ),
            packet,
        )
        assert [f.issue for f in found] == ["forbidden_strength"]
        assert found[0].section == "discussion"
        assert found[0].quote == "Statistically  Significant"
        assert found[0].origin == "deterministic"

    @pytest.mark.parametrize(
        "phrase",
        [
            "this proves the point",
            "a novel method",
            "state-of-the-art results",
            "we guarantee convergence",
            "an unprecedented breakthrough",
        ],
    )
    def test_each_lexicon_class_fires(self, phrase: str) -> None:
        packet = reportable_packet()
        found = deterministic_findings(
            manuscript_for(packet, abstract=phrase), packet
        )
        assert "forbidden_strength" in [f.issue for f in found]

    def test_word_boundaries_protect_innocent_words(self) -> None:
        packet = reportable_packet()
        found = deterministic_findings(
            manuscript_for(
                packet,
                abstract="training improves accuracy; results disprove "
                "nothing and take no approved shortcut",
            ),
            packet,
        )
        assert found == ()

    def test_an_unlicensed_verdict_word_fires(self) -> None:
        packet = reportable_packet()
        found = deterministic_findings(
            manuscript_for(packet, discussion="the null was refuted"),
            packet,
        )
        assert [f.issue for f in found] == ["unlicensed_verdict"]
        assert found[0].quote == "refuted"

    def test_a_licensed_verdict_word_passes(self) -> None:
        packet = reportable_packet()  # records "supported"
        found = deterministic_findings(
            manuscript_for(packet, abstract="the claim is supported"),
            packet,
        )
        assert found == ()

    def test_ordering_is_stable(self) -> None:
        packet = reportable_packet()
        draft = manuscript_for(
            packet,
            abstract="a novel result was refuted",
            discussion="we prove it",
        )
        first = deterministic_findings(draft, packet)
        assert first == deterministic_findings(draft, packet)
        assert [f.section for f in first] == [
            "abstract",
            "abstract",
            "discussion",
        ]


def model_finding(**overrides: str) -> dict[str, str]:
    entry = {
        "section": "discussion",
        "issue": "overclaim",
        "quote": "The claim is supported on held out data.",
        "subject_id": "",
        "explanation": "an explanation",
    }
    entry.update(overrides)
    return entry


class TestGrounding:
    def test_a_grounded_finding_survives(self) -> None:
        packet = reportable_packet()
        rejections, grounded = ground_findings(
            {"findings": [model_finding(subject_id=SOURCE_ID)]},
            manuscript_for(packet),
            packet,
        )
        assert rejections == ()
        assert len(grounded) == 1
        assert grounded[0].origin == "model"
        assert grounded[0].subject_id == SOURCE_ID

    def test_case_and_whitespace_variance_still_grounds(self) -> None:
        packet = reportable_packet()
        rejections, grounded = ground_findings(
            {
                "findings": [
                    model_finding(
                        quote="the claim IS  supported on held out data."
                    )
                ]
            },
            manuscript_for(packet),
            packet,
        )
        assert rejections == ()
        assert len(grounded) == 1

    def test_a_fabricated_quote_is_rejected(self) -> None:
        packet = reportable_packet()
        rejections, grounded = ground_findings(
            {"findings": [model_finding(quote="something never written")]},
            manuscript_for(packet),
            packet,
        )
        assert [r.rule for r in rejections] == ["ungrounded_quote"]
        assert grounded == ()

    def test_unknown_subject_section_and_issue_reject(self) -> None:
        packet = reportable_packet()
        rejections, grounded = ground_findings(
            {
                "findings": [
                    model_finding(
                        section="appendix",
                        issue="ugly_prose",
                        subject_id="lits_0000000000000000",
                    )
                ]
            },
            manuscript_for(packet),
            packet,
        )
        assert {r.rule for r in rejections} == {
            "unknown_section",
            "unknown_issue",
            "unknown_subject",
        }
        assert grounded == ()

    def test_an_empty_quote_is_rejected(self) -> None:
        packet = reportable_packet()
        rejections, _ = ground_findings(
            {"findings": [model_finding(quote="  ")]},
            manuscript_for(packet),
            packet,
        )
        assert "empty_quote" in [r.rule for r in rejections]

    def test_an_empty_payload_is_clean(self) -> None:
        packet = reportable_packet()
        assert ground_findings(
            {"findings": []}, manuscript_for(packet), packet
        ) == ((), ())


class TestVerdictAndSchema:
    def test_the_verdict_is_derived_never_output(self) -> None:
        assert derive_verdict(()) is ReviewVerdict.APPROVED
        finding = ReviewFinding(
            section="abstract",
            quote="q",
            issue="overclaim",
            subject_id="",
            explanation="e",
            origin="model",
        )
        assert derive_verdict((finding,)) is ReviewVerdict.REVISE

    def test_the_schema_has_no_verdict_and_pins_its_enums(self) -> None:
        packet = reportable_packet()
        schema = review_schema(packet)
        assert schema.name == "manuscript_review"
        properties = schema.json_schema["properties"]
        assert isinstance(properties, Mapping)
        assert set(properties) == {"findings"}
        item = properties["findings"]["items"]["properties"]
        assert tuple(item["section"]["enum"]) == PROSE_SECTIONS
        assert list(item["issue"]["enum"]) == sorted(MODEL_ISSUES)
        assert list(item["subject_id"]["enum"]) == [
            "",
            *sorted(packet_identifiers(packet)),
        ]

    def test_packet_identifiers_cover_both_renderings(self) -> None:
        ids = packet_identifiers(reportable_packet())
        assert SOURCE_ID in ids
        assert reportable_packet().packet_id in ids


def review_for(packet: EvidencePacket, manuscript: Manuscript) -> ReviewRecord:
    return ReviewRecord(
        manuscript_id=manuscript.manuscript_id,
        packet_id=packet.packet_id,
        verdict=ReviewVerdict.APPROVED,
        findings=(),
        call=manuscript.call,
    )


class TestRecords:
    def test_ids_derive_and_doctored_ids_refuse(self) -> None:
        packet = reportable_packet()
        review = review_for(packet, manuscript_for(packet))
        assert review.review_id.startswith("rvw_")
        with pytest.raises(ReviewError, match="does not survive"):
            replace(review, review_id="rvw_0000000000000000")
        revision = RevisionRecord(
            packet_id=packet.packet_id,
            review_id=review.review_id,
            superseded_manuscript_id="mscr_1",
            revision_manuscript_id="mscr_2",
        )
        assert revision.revision_id.startswith("rvn_")
        with pytest.raises(ReviewError, match="does not survive"):
            replace(revision, revision_id="rvn_0000000000000000")

    def test_an_unknown_issue_or_origin_refuses(self) -> None:
        with pytest.raises(ReviewError, match="unknown review issue"):
            ReviewFinding(
                section="abstract",
                quote="q",
                issue="ugly",
                subject_id="",
                explanation="e",
                origin="model",
            )
        with pytest.raises(ReviewError, match="unknown finding origin"):
            ReviewFinding(
                section="abstract",
                quote="q",
                issue="overclaim",
                subject_id="",
                explanation="e",
                origin="gremlin",
            )


class TestReviewStore:
    def test_round_trip_and_write_once(self, tmp_path: Path) -> None:
        store = ReviewStore(tmp_path)
        packet = reportable_packet()
        review = review_for(packet, manuscript_for(packet))
        store.record_review(review)
        assert store.get_review(review.review_id) == review
        assert store.record_review(review) == review  # no-op

    def test_a_doctored_file_fails_integrity(self, tmp_path: Path) -> None:
        store = ReviewStore(tmp_path)
        packet = reportable_packet()
        review = review_for(packet, manuscript_for(packet))
        store.record_review(review)
        path = tmp_path / f"{review.review_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["verdict"] = "revise"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ReviewIntegrityError):
            store.get_review(review.review_id)

    def test_lookups(self, tmp_path: Path) -> None:
        store = ReviewStore(tmp_path)
        packet = reportable_packet()
        manuscript = manuscript_for(packet)
        review = store.record_review(review_for(packet, manuscript))
        assert store.for_manuscript(manuscript.manuscript_id) == (review,)
        assert store.for_packet(packet.packet_id) == (review,)
        assert store.for_packet("epkt_0000000000000000") == ()
        revision = store.record_revision(
            RevisionRecord(
                packet_id=packet.packet_id,
                review_id=review.review_id,
                superseded_manuscript_id="mscr_1",
                revision_manuscript_id="mscr_2",
            )
        )
        assert store.revisions_for(packet.packet_id) == (revision,)

    def test_head_for(self, tmp_path: Path) -> None:
        manuscripts = ManuscriptStore(tmp_path / "m")
        reviews = ReviewStore(tmp_path / "r")
        packet = reportable_packet()
        first = manuscripts.record(manuscript_for(packet))
        assert head_for(manuscripts, reviews, packet.packet_id) == (first,)

        second = manuscripts.record(
            manuscript_for(packet, abstract="A different, revised draft.")
        )
        review = reviews.record_review(
            replace(
                review_for(packet, first),
                verdict=ReviewVerdict.REVISE,
                review_id="",
            )
        )
        reviews.record_revision(
            RevisionRecord(
                packet_id=packet.packet_id,
                review_id=review.review_id,
                superseded_manuscript_id=first.manuscript_id,
                revision_manuscript_id=second.manuscript_id,
            )
        )
        assert head_for(manuscripts, reviews, packet.packet_id) == (second,)

    def test_a_dangling_revision_is_an_integrity_error(
        self, tmp_path: Path
    ) -> None:
        manuscripts = ManuscriptStore(tmp_path / "m")
        reviews = ReviewStore(tmp_path / "r")
        packet = reportable_packet()
        manuscripts.record(manuscript_for(packet))
        reviews.record_revision(
            RevisionRecord(
                packet_id=packet.packet_id,
                review_id="rvw_1",
                superseded_manuscript_id="mscr_1",
                revision_manuscript_id="mscr_nowhere",
            )
        )
        with pytest.raises(ReviewIntegrityError, match="does not hold"):
            head_for(manuscripts, reviews, packet.packet_id)


def reviewer_with(
    replies: list[str], tmp_path: Path, *, max_corrective_calls: int = 1
) -> tuple[FaithfulnessReviewer, FakeModelProvider, ReviewStore]:
    provider = FakeModelProvider(replies)
    store = ReviewStore(tmp_path / "review")
    return (
        FaithfulnessReviewer(
            provider=provider,
            model="fake-model-1",
            ledger=UsageLedger(),
            store=store,
            max_corrective_calls=max_corrective_calls,
        ),
        provider,
        store,
    )


EMPTY_REVIEW = json.dumps({"findings": []})
GROUNDED_REVIEW = json.dumps({"findings": [model_finding()]})
UNGROUNDED_REVIEW = json.dumps(
    {"findings": [model_finding(quote="text never written")]}
)


class TestFaithfulnessReviewer:
    def test_a_faithful_draft_approves_in_one_call(
        self, tmp_path: Path
    ) -> None:
        packet = reportable_packet()
        reviewer, provider, store = reviewer_with([EMPTY_REVIEW], tmp_path)
        review = reviewer.review(packet, manuscript_for(packet))

        assert review.verdict is ReviewVerdict.APPROVED
        assert review.findings == ()
        assert len(provider.calls) == 1
        assert review.call.repair_count == 0
        assert store.rejected() == ()

    def test_a_grounded_finding_becomes_revise(self, tmp_path: Path) -> None:
        packet = reportable_packet()
        reviewer, _, _ = reviewer_with([GROUNDED_REVIEW], tmp_path)
        review = reviewer.review(packet, manuscript_for(packet))

        assert review.verdict is ReviewVerdict.REVISE
        assert [f.origin for f in review.findings] == ["model"]

    def test_deterministic_and_model_findings_merge(
        self, tmp_path: Path
    ) -> None:
        packet = reportable_packet()
        draft = manuscript_for(packet, limitations="a novel limitation")
        reviewer, _, _ = reviewer_with([GROUNDED_REVIEW], tmp_path)
        review = reviewer.review(packet, draft)

        assert [f.origin for f in review.findings] == [
            "deterministic",
            "model",
        ]

    def test_an_ungrounded_review_earns_one_corrective(
        self, tmp_path: Path
    ) -> None:
        packet = reportable_packet()
        reviewer, provider, store = reviewer_with(
            [UNGROUNDED_REVIEW, EMPTY_REVIEW], tmp_path
        )
        review = reviewer.review(packet, manuscript_for(packet))

        assert review.verdict is ReviewVerdict.APPROVED
        assert review.call.repair_count == 1
        assert len(store.rejected()) == 1
        corrective = provider.calls[1].messages[-1].content
        assert "ungrounded_quote" in corrective

    def test_exhausted_correctives_are_a_typed_refusal(
        self, tmp_path: Path
    ) -> None:
        packet = reportable_packet()
        reviewer, provider, store = reviewer_with(
            [UNGROUNDED_REVIEW, UNGROUNDED_REVIEW], tmp_path
        )
        with pytest.raises(ReviewRejectedError, match="ground itself"):
            reviewer.review(packet, manuscript_for(packet))
        assert len(provider.calls) == 2
        assert len(store.rejected()) == 2

    def test_non_json_replies_follow_the_same_discipline(
        self, tmp_path: Path
    ) -> None:
        packet = reportable_packet()
        reviewer, _, _ = reviewer_with(["nope", EMPTY_REVIEW], tmp_path)
        review = reviewer.review(packet, manuscript_for(packet))
        assert review.call.repair_count == 1

        reviewer, _, _ = reviewer_with(["nope", "still nope"], tmp_path / "b")
        with pytest.raises(StructuredOutputError):
            reviewer.review(packet, manuscript_for(packet))

    def test_a_mismatched_manuscript_refuses_before_any_call(
        self, tmp_path: Path
    ) -> None:
        packet = reportable_packet()
        other = replace(
            packet,
            science=replace(packet.science, question="another?"),
            packet_id="",
        )
        reviewer, provider, _ = reviewer_with([EMPTY_REVIEW], tmp_path)
        with pytest.raises(Exception, match="was authored from"):
            reviewer.review(other, manuscript_for(packet))
        assert provider.calls == ()

    def test_gate_drift_refuses_before_any_call(self, tmp_path: Path) -> None:
        packet = reportable_packet()
        drifted = manuscript_for(packet, abstract="an invented 99.9 number")
        reviewer, provider, _ = reviewer_with([EMPTY_REVIEW], tmp_path)
        with pytest.raises(ReviewError, match="no longer passes"):
            reviewer.review(packet, drifted)
        assert provider.calls == ()


# -- the cycle over a stub lab ------------------------------------------------


class _FakeLab:
    def __init__(self, replies: list[str]) -> None:
        self.provider = FakeModelProvider(replies)

    def model_provider(self, stage: StageName, /) -> FakeModelProvider:
        del stage
        return self.provider

    def literature_provider(self) -> NoReturn:
        raise AssertionError("the review path asked for literature")

    def runtime(self, request: object, /) -> NoReturn:
        raise AssertionError("the review path asked for a runtime")


@pytest.fixture(scope="module")
def walked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("review") / "run"
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    return root


def authored(root: Path) -> None:
    from autonomous_research_lab.control.manuscript import author_manuscript

    author_manuscript(root, lab=ci_lab())


def draft_prose_reply(root: Path, **overrides: str) -> str:
    """A reply that revises the standing draft: the fixture's own
    sections with overrides, so gates keep passing."""
    from autonomous_research_lab.control.packet import build_packet

    packet = build_packet(root)
    store = ManuscriptStore(root / "manuscript")
    manuscript = store.for_packet(packet.packet_id)[0]
    sections = {
        name: getattr(manuscript.sections, name) for name in PROSE_SECTIONS
    }
    sections.update(overrides)
    return json.dumps(sections)


def grounded_reply(root: Path) -> str:
    """A model finding grounded in the standing draft's abstract."""
    from autonomous_research_lab.control.packet import build_packet

    packet = build_packet(root)
    store = ManuscriptStore(root / "manuscript")
    manuscript = store.for_packet(packet.packet_id)[0]
    quote = manuscript.sections.abstract.split(".")[0]
    return json.dumps(
        {
            "findings": [
                {
                    "section": "abstract",
                    "issue": "overclaim",
                    "quote": quote,
                    "subject_id": "",
                    "explanation": "claims more than the record states",
                }
            ]
        }
    )


class TestReviewCycle:
    def test_the_full_cycle_ends_approved(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        authored(root)

        lab = _FakeLab(
            [
                grounded_reply(root),  # r1: REVISE
                draft_prose_reply(
                    root, abstract="A revised, faithful abstract."
                ),  # m2
                EMPTY_REVIEW,  # r2: APPROVED
            ]
        )
        outcome = review_manuscript(root, lab=lab)

        assert outcome.review.verdict is ReviewVerdict.APPROVED
        assert outcome.opening_review is not None
        assert outcome.opening_review.verdict is ReviewVerdict.REVISE
        assert outcome.superseded_manuscript is not None
        assert (
            outcome.manuscript.sections.abstract
            == "A revised, faithful abstract."
        )
        # The revision request carried the findings.
        revision_request = lab.provider.calls[1]
        assert "overclaim" in revision_request.messages[-1].content
        assert (
            revision_request.metadata["revision_of"]
            == outcome.superseded_manuscript.manuscript_id
        )
        # Succession is durable; the head is the revision.
        reviews = ReviewStore(root / "review")
        manuscripts = ManuscriptStore(root / "manuscript")
        packet_id = outcome.packet.packet_id
        assert len(reviews.revisions_for(packet_id)) == 1
        assert head_for(manuscripts, reviews, packet_id) == (
            outcome.manuscript,
        )

    def test_a_standing_revise_replays_and_never_authors_again(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        authored(root)

        lab = _FakeLab(
            [
                grounded_reply(root),
                draft_prose_reply(root, abstract="A revised abstract."),
                # r2 also finds an issue, grounded in the revised draft:
                json.dumps(
                    {
                        "findings": [
                            {
                                "section": "abstract",
                                "issue": "overclaim",
                                "quote": "A revised abstract.",
                                "subject_id": "",
                                "explanation": "still overclaims",
                            }
                        ]
                    }
                ),
            ]
        )
        first = review_manuscript(root, lab=lab)
        assert first.review.verdict is ReviewVerdict.REVISE
        assert not first.replayed

        again = review_manuscript(root, lab=_FakeLab([]))
        assert again.replayed  # zero calls: the fake would raise on any
        assert again.review.review_id == first.review.review_id

    def test_review_only_stops_after_the_first_review(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        authored(root)

        lab = _FakeLab([grounded_reply(root)])
        first = review_manuscript(root, lab=lab, revise=False)
        assert first.review.verdict is ReviewVerdict.REVISE
        assert first.opening_review is None
        assert ReviewStore(root / "review").revisions_for(
            first.packet.packet_id
        ) == ()

        # A later full run resumes from the recorded findings: it authors
        # and re-reviews without repeating the first review call.
        resume = _FakeLab(
            [
                draft_prose_reply(root, abstract="A calmer abstract."),
                EMPTY_REVIEW,
            ]
        )
        finished = review_manuscript(root, lab=resume)
        assert finished.review.verdict is ReviewVerdict.APPROVED
        assert finished.opening_review is not None
        assert (
            finished.opening_review.review_id == first.review.review_id
        )

    def test_the_adoption_rule_recovers_a_missing_succession(
        self, tmp_path: Path
    ) -> None:
        """Crash window W2: the revision draft was recorded but the
        succession was not. The re-run adopts it and continues."""
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        authored(root)

        opening = _FakeLab([grounded_reply(root)])
        first = review_manuscript(root, lab=opening, revise=False)

        # Plant the orphaned revision by hand: a second draft, no rvn.
        manuscripts = ManuscriptStore(root / "manuscript")
        head = manuscripts.for_packet(first.packet.packet_id)[0]
        orphan = manuscripts.record(
            replace(
                head,
                sections=replace(
                    head.sections, abstract="An orphaned revision."
                ),
                manuscript_id="",
            )
        )

        finish = _FakeLab([EMPTY_REVIEW])
        outcome = review_manuscript(root, lab=finish)
        assert outcome.manuscript.manuscript_id == orphan.manuscript_id
        assert outcome.review.verdict is ReviewVerdict.APPROVED
        assert outcome.opening_review is not None

    def test_an_unrecognizable_multi_head_state_refuses(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        authored(root)

        # Two extra drafts with no review at all: not a lab state.
        manuscripts = ManuscriptStore(root / "manuscript")
        from autonomous_research_lab.control.packet import build_packet

        packet = build_packet(root)
        head = manuscripts.for_packet(packet.packet_id)[0]
        for text in ("A stray draft.", "Another stray draft."):
            manuscripts.record(
                replace(
                    head,
                    sections=replace(head.sections, abstract=text),
                    manuscript_id="",
                )
            )

        with pytest.raises(AmbiguousHeadError, match="drafts stand"):
            review_manuscript(root, lab=_FakeLab([]))


class TestReviewVerbIntegration:
    def test_the_scripted_lab_approves_its_own_draft(
        self, walked: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        authored(walked)
        code = main(
            [
                "review",
                "--root",
                str(walked),
                "--lab",
                "examples.vision_lab:ci_lab",
            ]
        )
        printed = capsys.readouterr().out
        assert code == OK
        assert "verdict       approved" in printed
        assert main(["verify", "--root", str(walked)]) == OK

    def test_a_second_run_replays(self, walked: Path) -> None:
        authored(walked)
        first = review_manuscript(walked, lab=ci_lab())
        again = review_manuscript(walked, lab=_FakeLab([]))
        assert again.replayed
        assert again.review.review_id == first.review.review_id

    def test_no_manuscript_is_a_refusal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok

        code = main(
            [
                "review",
                "--root",
                str(root),
                "--lab",
                "examples.vision_lab:ci_lab",
            ]
        )
        printed = capsys.readouterr().out
        assert code == REFUSED
        assert "author first" in printed

    def test_a_walk_with_nothing_to_report_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab(), stop_after=StageName.ADMISSION)
        assert result.ok

        code = main(["review", "--root", str(root)])
        assert code == REFUSED
        assert "REFUSED" in capsys.readouterr().out
