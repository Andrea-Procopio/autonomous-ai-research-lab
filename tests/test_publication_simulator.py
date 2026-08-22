"""The venue simulator: an instrument that reads, never decides.

Unit half: the schema (no verdict property, integer enums), trusted
aggregation, the three record kinds, the store, and the seat over the
scripted fake provider. Composition half: the full below-bar polish
cycle over a stub lab, every crash window's recovery, the bounded-cycle
invariant, and the head-resolution union that keeps author, review, and
render agreeing on which draft stands.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from autonomous_research_lab.control.cli import FAILED, OK, REFUSED, main
from autonomous_research_lab.control.manuscript import author_manuscript
from autonomous_research_lab.control.render import render_submission
from autonomous_research_lab.control.review import review_manuscript
from autonomous_research_lab.control.simulate import (
    PolishRejectedError,
    simulate_submission,
)
from autonomous_research_lab.control.stage import StageName
from autonomous_research_lab.publication.manuscript import (
    AuthorCall,
)
from autonomous_research_lab.publication.simulator import (
    DIMENSIONS,
    LENSES,
    PolishRecord,
    SimulationError,
    SimulationRecord,
    SimulationRejectedError,
    VenueReview,
    VenueSimulator,
    aggregate,
    meets,
    review_form_schema,
)
from autonomous_research_lab.publication.store import (
    ManuscriptStore,
    ReviewStore,
    SimulationIntegrityError,
    SimulationStore,
    head_for,
)
from autonomous_research_lab.runtime.providers import (
    FakeModelProvider,
    OutputSchema,
    UsageLedger,
)
from examples.vision_chain import walk
from examples.vision_lab import ci_lab

TEX_SHA = "a" * 64


def form(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "summary": "A compact, honest study.",
        "strengths": ["traceable figures"],
        "weaknesses": [],
        "questions": [],
        **{name: 3 for name in DIMENSIONS},
        "overall": 7,
        "confidence": 4,
    }
    payload.update(overrides)
    return payload


def venue_review(**overrides: object) -> VenueReview:
    fields: dict[str, object] = {
        "manuscript_id": "mscr_1",
        "packet_id": "epkt_1",
        "venue_name": "plain",
        "tex_sha256": TEX_SHA,
        "lens": "rigor",
        "summary": "A compact, honest study.",
        "strengths": ("traceable figures",),
        "weaknesses": (),
        "questions": (),
        **{name: 3 for name in DIMENSIONS},
        "overall": 7,
        "confidence": 4,
        "call": AuthorCall(
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
    }
    fields.update(overrides)
    return VenueReview(**fields)  # type: ignore[arg-type]


class TestSchema:
    def test_the_form_has_no_verdict_property(self) -> None:
        schema = review_form_schema()
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert "verdict" not in properties
        assert "decision" not in properties
        assert set(properties) == {
            "summary",
            "strengths",
            "weaknesses",
            "questions",
            *DIMENSIONS,
            "overall",
            "confidence",
        }

    def test_scores_are_integer_enums(self) -> None:
        schema = OutputSchema(
            name="venue_review", json_schema=review_form_schema()
        )
        assert schema.validate(form()) is not None
        with pytest.raises(Exception, match="does not satisfy"):
            schema.validate(form(overall=11))
        with pytest.raises(Exception, match="does not satisfy"):
            schema.validate(form(overall=True))
        with pytest.raises(Exception, match="does not satisfy"):
            schema.validate(form(quality="3"))

    def test_a_filled_integer_form_round_trips_the_provider(self) -> None:
        from autonomous_research_lab.runtime.providers import (
            Message,
            MessageRole,
            ModelRequest,
        )

        provider = FakeModelProvider([json.dumps(form())])
        response = provider.invoke(
            ModelRequest(
                model="fake-model-1",
                messages=(
                    Message(role=MessageRole.USER, content="the paper"),
                ),
                schema=OutputSchema(
                    name="venue_review", json_schema=review_form_schema()
                ),
            )
        )
        assert response.structured is not None
        assert response.structured["overall"] == 7


class TestAggregation:
    def test_medians_are_floats_odd_and_even(self) -> None:
        odd = aggregate(
            (
                venue_review(overall=5),
                venue_review(lens="clarity", overall=7),
                venue_review(lens="significance", overall=9),
            )
        )
        assert dict(odd)["overall"] == 7.0
        assert isinstance(dict(odd)["overall"], float)
        even = aggregate(
            (venue_review(overall=6), venue_review(lens="clarity", overall=7))
        )
        assert dict(even)["overall"] == 6.5

    def test_the_bar_boundary(self) -> None:
        assert meets(6, 6.0)
        assert not meets(6, 5.5)

    def test_zero_reviews_refuse(self) -> None:
        with pytest.raises(SimulationError, match="zero reviews"):
            aggregate(())


class TestRecords:
    def test_ids_derive_and_doctored_ids_refuse(self) -> None:
        review = venue_review()
        assert review.review_id.startswith("vrev_")
        with pytest.raises(SimulationError, match="does not survive"):
            replace(review, review_id="vrev_0000000000000000")

        simulation = SimulationRecord(
            manuscript_id="mscr_1",
            packet_id="epkt_1",
            venue_name="plain",
            tex_sha256=TEX_SHA,
            bar=6,
            review_ids=(review.review_id,),
            medians=(("overall", 7.0),),
            meets_bar=True,
        )
        assert simulation.simulation_id.startswith("vsim_")
        with pytest.raises(SimulationError, match="does not survive"):
            replace(simulation, simulation_id="vsim_0000000000000000")

        polish = PolishRecord(
            packet_id="epkt_1",
            simulation_id=simulation.simulation_id,
            superseded_manuscript_id="mscr_1",
            revision_manuscript_id="mscr_2",
        )
        assert polish.polish_id.startswith("plsh_")
        with pytest.raises(SimulationError, match="does not survive"):
            replace(polish, polish_id="plsh_0000000000000000")


class TestSimulationStore:
    def test_round_trip_and_write_once(self, tmp_path: Path) -> None:
        store = SimulationStore(tmp_path)
        review = store.record_review(venue_review())
        assert store.get_review(review.review_id) == review
        assert store.record_review(review) == review
        assert store.reviews_for("mscr_1", TEX_SHA) == (review,)
        assert store.reviews_for("mscr_1", "b" * 64) == ()

    def test_a_doctored_file_fails_integrity(self, tmp_path: Path) -> None:
        store = SimulationStore(tmp_path)
        review = store.record_review(venue_review())
        path = tmp_path / f"{review.review_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["overall"] = 10
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SimulationIntegrityError):
            store.get_review(review.review_id)

    def test_polish_lookup(self, tmp_path: Path) -> None:
        store = SimulationStore(tmp_path)
        polish = store.record_polish(
            PolishRecord(
                packet_id="epkt_1",
                simulation_id="vsim_1",
                superseded_manuscript_id="mscr_1",
                revision_manuscript_id="mscr_2",
            )
        )
        assert store.polishes_for("epkt_1") == (polish,)
        assert store.polishes_for("epkt_other") == ()


def simulator_with(
    replies: list[str], tmp_path: Path
) -> tuple[VenueSimulator, FakeModelProvider, SimulationStore]:
    provider = FakeModelProvider(replies)
    store = SimulationStore(tmp_path / "simulation")
    return (
        VenueSimulator(
            provider=provider,
            model="fake-model-1",
            ledger=UsageLedger(),
            store=store,
        ),
        provider,
        store,
    )


class TestSeat:
    def test_one_call_with_lens_and_venue_in_the_instruction(
        self, tmp_path: Path
    ) -> None:
        simulator, provider, _ = simulator_with(
            [json.dumps(form())], tmp_path
        )
        review = simulator.review(
            main_tex="\\documentclass{article}",
            references_bib="@misc{x, title={{T}}}",
            venue_name="plain",
            anonymous=False,
            lens="rigor",
            focus=LENSES[0][1],
            manuscript_id="mscr_1",
            packet_id="epkt_1",
            tex_sha256=TEX_SHA,
        )
        assert len(provider.calls) == 1
        request = provider.calls[0]
        assert "reviewer for plain" in request.instruction
        assert LENSES[0][1] in request.instruction
        assert "double-blind" not in request.instruction
        assert request.temperature == 0.0
        assert "=== main.tex ===" in request.messages[0].content
        assert review.tex_sha256 == TEX_SHA
        assert review.call.provider == "fake"
        assert review.overall == 7

    def test_anonymity_is_stated_for_blind_venues(
        self, tmp_path: Path
    ) -> None:
        simulator, provider, _ = simulator_with(
            [json.dumps(form())], tmp_path
        )
        simulator.review(
            main_tex="x",
            references_bib="",
            venue_name="neurips",
            anonymous=True,
            lens="clarity",
            focus="f",
            manuscript_id="mscr_1",
            packet_id="epkt_1",
            tex_sha256=TEX_SHA,
        )
        assert "double-blind" in provider.calls[0].instruction

    def test_an_empty_summary_earns_a_corrective_then_refuses(
        self, tmp_path: Path
    ) -> None:
        simulator, _provider, store = simulator_with(
            [json.dumps(form(summary="  ")), json.dumps(form())], tmp_path
        )
        review = simulator.review(
            main_tex="x",
            references_bib="",
            venue_name="plain",
            anonymous=False,
            lens="rigor",
            focus="f",
            manuscript_id="mscr_1",
            packet_id="epkt_1",
            tex_sha256=TEX_SHA,
        )
        assert review.call.repair_count == 1
        assert len(store.rejected()) == 1
        assert store.rejected()[0]["lens"] == "rigor"

        simulator, _provider, store = simulator_with(
            [json.dumps(form(summary="")), json.dumps(form(summary=""))],
            tmp_path / "b",
        )
        with pytest.raises(SimulationRejectedError, match="empty_summary"):
            simulator.review(
                main_tex="x",
                references_bib="",
                venue_name="plain",
                anonymous=False,
                lens="rigor",
                focus="f",
                manuscript_id="mscr_1",
                packet_id="epkt_1",
                tex_sha256=TEX_SHA,
            )


# -- composition over a walked root -------------------------------------------


class _FakeLab:
    def __init__(self, replies: list[str]) -> None:
        self.provider = FakeModelProvider(replies)

    def model_provider(self, stage: StageName, /) -> FakeModelProvider:
        del stage
        return self.provider

    def literature_provider(self) -> NoReturn:
        raise AssertionError("the simulate path asked for literature")

    def runtime(self, request: object, /) -> NoReturn:
        raise AssertionError("the simulate path asked for a runtime")


LOW_FORM = json.dumps(
    form(
        overall=4,
        weaknesses=["the introduction underexplains the setup"],
    )
)
HIGH_FORM = json.dumps(form())
CLEAN_REVIEW = json.dumps({"findings": []})


def prepared(root: Path) -> None:
    """A walked root with an approved draft."""
    result = walk(root, lab=ci_lab())
    assert result.ok, result.detail
    author_manuscript(root, lab=ci_lab())
    outcome = review_manuscript(root, lab=ci_lab())
    assert str(outcome.review.verdict) == "approved"


def revised_prose(root: Path, **overrides: str) -> str:
    from autonomous_research_lab.control.packet import build_packet
    from autonomous_research_lab.publication.manuscript import (
        PROSE_SECTIONS,
    )

    packet = build_packet(root)
    store = ManuscriptStore(root / "manuscript")
    manuscript = store.for_packet(packet.packet_id)[0]
    sections = {
        name: getattr(manuscript.sections, name) for name in PROSE_SECTIONS
    }
    sections.update(overrides)
    return json.dumps(sections)


@pytest.fixture(scope="module")
def approved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("simulate") / "run"
    prepared(root)
    return root


class TestSimulateMeetsBar:
    def test_the_scripted_lab_meets_the_default_bar(
        self, approved: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "simulate",
                "--root",
                str(approved),
                "--venue",
                "plain",
                "--lab",
                "examples.vision_lab:ci_lab",
            ]
        )
        printed = capsys.readouterr().out
        assert code == OK
        assert "MEETS BAR" in printed
        assert "rigor" in printed and "significance" in printed
        store = SimulationStore(approved / "simulation")
        assert len(store.reviews()) == 3
        assert len(store.simulations()) == 1
        assert main(["verify", "--root", str(approved)]) == OK

    def test_a_second_run_replays_with_zero_calls(
        self, approved: Path
    ) -> None:
        result = simulate_submission(
            approved,
            venue="plain",
            venue_config=None,
            kits_root=None,
            lab=_FakeLab([]),
        )
        assert result.replayed
        assert result.simulation.meets_bar

    def test_a_new_bar_reuses_recorded_reviews(self, approved: Path) -> None:
        result = simulate_submission(
            approved,
            venue="plain",
            venue_config=None,
            kits_root=None,
            lab=_FakeLab([]),
            bar=8,
            polish=False,
        )
        assert not result.simulation.meets_bar  # median 7.0 < 8
        assert result.simulation.bar == 8
        # zero model calls: the fake would have raised on any


class TestSimulatePolishCycle:
    def test_below_bar_runs_one_polish_cycle(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        prepared(root)
        lab = _FakeLab(
            [
                LOW_FORM,
                LOW_FORM,
                LOW_FORM,  # ensemble 1: below bar
                revised_prose(root, introduction="A clearer introduction."),
                CLEAN_REVIEW,  # faithfulness of the polish
                HIGH_FORM,
                HIGH_FORM,
                HIGH_FORM,  # ensemble 2
            ]
        )
        result = simulate_submission(
            root, venue="plain", venue_config=None, kits_root=None, lab=lab
        )

        assert result.simulation.meets_bar
        assert result.opening_simulation is not None
        assert not result.opening_simulation.meets_bar
        assert result.polish is not None
        assert result.faithfulness is not None
        assert str(result.faithfulness.verdict) == "approved"
        # The polish request carried the lens-labeled weakness and the
        # polish preamble, not the faithfulness one.
        polish_request = lab.provider.calls[3]
        prompt = polish_request.messages[-1].content
        assert "presentation weaknesses" in prompt
        assert "rigor: the introduction underexplains" in prompt
        assert polish_request.metadata.get("polish_of")
        # Succession is durable and every verb sees the polish head.
        manuscripts = ManuscriptStore(root / "manuscript")
        reviews = ReviewStore(root / "review")
        simulations = SimulationStore(root / "simulation")
        heads = head_for(
            manuscripts, reviews, result.simulation.packet_id, simulations
        )
        assert [found.manuscript_id for found in heads] == [
            result.simulation.manuscript_id
        ]
        assert main(["verify", "--root", str(root)]) == OK

    def test_the_polish_cycle_is_bounded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        prepared(root)
        lab = _FakeLab(
            [
                LOW_FORM,
                LOW_FORM,
                LOW_FORM,
                revised_prose(root, introduction="Still modest."),
                CLEAN_REVIEW,
                LOW_FORM,
                LOW_FORM,
                LOW_FORM,  # still below bar
            ]
        )
        result = simulate_submission(
            root, venue="plain", venue_config=None, kits_root=None, lab=lab
        )
        assert not result.simulation.meets_bar

        # Re-run: no second polish, standing verdict replays.
        again = simulate_submission(
            root,
            venue="plain",
            venue_config=None,
            kits_root=None,
            lab=_FakeLab([]),
        )
        assert again.replayed
        assert not again.simulation.meets_bar
        assert (
            len(
                SimulationStore(root / "simulation").polishes_for(
                    result.simulation.packet_id
                )
            )
            == 1
        )
        code = main(
            [
                "simulate",
                "--root",
                str(root),
                "--venue",
                "plain",
                "--lab",
                "examples.vision_lab:ci_lab",
            ]
        )
        printed = capsys.readouterr().out
        assert code == FAILED
        assert "BELOW BAR" in printed
        assert "underexplains" in printed

    def test_revise_after_polish_is_a_typed_stop(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        prepared(root)
        grounded = json.dumps(
            {
                "findings": [
                    {
                        "section": "introduction",
                        "issue": "overclaim",
                        "quote": "A bolder introduction",
                        "subject_id": "",
                        "explanation": "claims more than the record",
                    }
                ]
            }
        )
        lab = _FakeLab(
            [
                LOW_FORM,
                LOW_FORM,
                LOW_FORM,
                revised_prose(root, introduction="A bolder introduction."),
                grounded,  # faithfulness REVISE on the polish
            ]
        )
        with pytest.raises(PolishRejectedError, match="does not outrank"):
            simulate_submission(
                root, venue="plain", venue_config=None, kits_root=None, lab=lab
            )
        # Idempotent: re-run replays the standing REVISE, zero calls.
        with pytest.raises(PolishRejectedError):
            simulate_submission(
                root,
                venue="plain",
                venue_config=None,
                kits_root=None,
                lab=_FakeLab([]),
            )

    def test_an_orphan_polish_draft_is_adopted(self, tmp_path: Path) -> None:
        """Crash window W4: the polish draft exists but its succession
        record does not."""
        root = tmp_path / "run"
        prepared(root)
        lab = _FakeLab([LOW_FORM, LOW_FORM, LOW_FORM])
        first = simulate_submission(
            root,
            venue="plain",
            venue_config=None,
            kits_root=None,
            lab=lab,
            polish=False,
        )
        assert not first.simulation.meets_bar

        # Plant the orphan by hand: the polish draft with no plsh.

        manuscripts = ManuscriptStore(root / "manuscript")
        head = manuscripts.for_packet(first.simulation.packet_id)[0]
        orphan = manuscripts.record(
            replace(
                head,
                sections=replace(
                    head.sections, introduction="An orphaned polish."
                ),
                manuscript_id="",
            )
        )

        finish = _FakeLab([CLEAN_REVIEW, HIGH_FORM, HIGH_FORM, HIGH_FORM])
        outcome = simulate_submission(
            root, venue="plain", venue_config=None, kits_root=None, lab=finish
        )
        assert outcome.simulation.manuscript_id == orphan.manuscript_id
        assert outcome.simulation.meets_bar

    def test_mid_ensemble_crash_replays_recorded_lenses_only(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        prepared(root)
        # First run records one lens then dies (fake runs out of replies).
        starving = _FakeLab([HIGH_FORM])
        with pytest.raises(Exception, match="scripted for 1 call"):
            simulate_submission(
                root,
                venue="plain",
                venue_config=None,
                kits_root=None,
                lab=starving,
            )
        assert len(SimulationStore(root / "simulation").reviews()) == 1

        resume = _FakeLab([HIGH_FORM, HIGH_FORM])  # only the two owed
        outcome = simulate_submission(
            root, venue="plain", venue_config=None, kits_root=None, lab=resume
        )
        assert outcome.simulation.meets_bar
        assert len(resume.provider.calls) == 2

    def test_no_polish_stops_after_the_first_reading(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "run"
        prepared(root)
        lab = _FakeLab([LOW_FORM, LOW_FORM, LOW_FORM])
        result = simulate_submission(
            root,
            venue="plain",
            venue_config=None,
            kits_root=None,
            lab=lab,
            polish=False,
        )
        assert not result.simulation.meets_bar
        assert result.polish is None
        assert (
            SimulationStore(root / "simulation").polishes_for(
                result.simulation.packet_id
            )
            == ()
        )

    def test_reviews_out_of_range_refuse(self, approved: Path) -> None:
        with pytest.raises(SimulationError, match="reviews must be"):
            simulate_submission(
                approved,
                venue="plain",
                venue_config=None,
                kits_root=None,
                reviews=4,
            )
        with pytest.raises(SimulationError, match="bar must be"):
            simulate_submission(
                approved,
                venue="plain",
                venue_config=None,
                kits_root=None,
                bar=0,
            )


class TestSimulateRefusals:
    def test_an_unapproved_draft_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok
        author_manuscript(root, lab=ci_lab())

        code = main(["simulate", "--root", str(root), "--venue", "plain"])
        assert code == REFUSED
        assert "review first" in capsys.readouterr().out

    def test_no_manuscript_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "run"
        result = walk(root, lab=ci_lab())
        assert result.ok

        code = main(["simulate", "--root", str(root), "--venue", "plain"])
        assert code == REFUSED
        assert "author first" in capsys.readouterr().out


class TestHeadUnionAcrossVerbs:
    def test_every_verb_sees_the_polish_head(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        prepared(root)
        lab = _FakeLab(
            [
                LOW_FORM,
                LOW_FORM,
                LOW_FORM,
                revised_prose(root, introduction="Polished."),
                CLEAN_REVIEW,
                HIGH_FORM,
                HIGH_FORM,
                HIGH_FORM,
            ]
        )
        outcome = simulate_submission(
            root, venue="plain", venue_config=None, kits_root=None, lab=lab
        )
        polished_id = outcome.simulation.manuscript_id

        replayed = author_manuscript(root, lab=_FakeLab([]))
        assert replayed.manuscript.manuscript_id == polished_id
        reviewed = review_manuscript(root, lab=_FakeLab([]))
        assert reviewed.manuscript.manuscript_id == polished_id
        rendered = render_submission(
            root, venue="plain", venue_config=None, kits_root=None
        )
        assert rendered.manuscript.manuscript_id == polished_id

    def test_a_dangling_polish_is_an_integrity_error(
        self, tmp_path: Path
    ) -> None:
        manuscripts = ManuscriptStore(tmp_path / "m")
        reviews = ReviewStore(tmp_path / "r")
        simulations = SimulationStore(tmp_path / "s")
        simulations.record_polish(
            PolishRecord(
                packet_id="epkt_1",
                simulation_id="vsim_1",
                superseded_manuscript_id="mscr_1",
                revision_manuscript_id="mscr_nowhere",
            )
        )
        with pytest.raises(SimulationIntegrityError, match="does not hold"):
            head_for(manuscripts, reviews, "epkt_1", simulations)

    def test_head_for_without_simulations_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        manuscripts = ManuscriptStore(tmp_path / "m")
        reviews = ReviewStore(tmp_path / "r")
        assert head_for(manuscripts, reviews, "epkt_1") == ()
