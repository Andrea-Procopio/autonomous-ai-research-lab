"""The vision lab: grammar, catalog, fixed regions, and the whole walk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.admission.admitter import encoded_metric
from autonomous_research_lab.control.chain import ExperimentationStage
from autonomous_research_lab.control.lab import (
    ExperimentationUnavailableError,
    RuntimeRequest,
)
from autonomous_research_lab.core.hypothesis import Hypothesis
from autonomous_research_lab.core.prediction import Comparator, Prediction
from autonomous_research_lab.core.question import ResearchQuestion
from autonomous_research_lab.evidence.file_store import FileEvidenceStore
from autonomous_research_lab.persistence.commit_store import CommitBundleStore
from autonomous_research_lab.persistence.state_store import FileStateStore
from autonomous_research_lab.program.integrity import verify_run
from autonomous_research_lab.program.journal import RunJournal
from autonomous_research_lab.program.ledger import BudgetLedger
from autonomous_research_lab.roles.base import RoleContext
from autonomous_research_lab.roles.planner import (
    TemplateCatalog,
    check_decision,
)
from examples.vision_chain import walk
from examples.vision_lab import ci_lab
from examples.vision_lab.catalog import (
    PLACEHOLDER,
    catalog_for,
    fill_slot,
    fixed_regions,
)
from examples.vision_lab.composition import STUB_SLOT, _funded_state
from examples.vision_lab.measure import (
    SUPPORTED_CONTRASTS,
    UnmeasurablePredictionsError,
    parse_contrast,
    require_measurable,
    template_for,
)

ENCODER_METRIC = (
    "difference in linear probe accuracy: "
    "trained encoder minus randomly initialized encoder"
)

#: The two metric strings the preserved Task 5F admission actually
#: carries — the honest-refusal fixture, inlined so the test needs no
#: live_runs directory.
FIVE_F_METRICS = (
    "difference in overlap of important heads: "
    "top-weighted heads minus bottom-weighted heads",
    "difference in few-shot accuracy: "
    "bottom-weighted heads minus top-weighted heads",
)


def admitted(metric: str) -> Prediction:
    return Prediction(
        hypothesis_id="hyp_1",
        condition="on held-out images from the evaluation split",
        metric=metric,
        comparator=Comparator.GREATER_THAN,
        threshold=0.0,
        expectation="the trained arm stays above the untrained arm",
    )


class TestTheMetricGrammar:
    def test_it_round_trips_what_admission_encodes(self) -> None:
        """The parser is a mirror of ``encoded_metric``; this is the test
        that holds the two together."""

        class Entry:
            base_metric = "linear probe accuracy"
            expected_higher_arm = "trained encoder"
            expected_lower_arm = "randomly initialized encoder"

        rendered = encoded_metric(Entry())  # type: ignore[arg-type]
        contrast = parse_contrast(rendered)

        assert contrast is not None
        assert contrast.key in SUPPORTED_CONTRASTS
        assert template_for(rendered) == "vision-encoder-contrast-v1"

    def test_unparseable_strings_are_unmeasurable(self) -> None:
        assert parse_contrast("accuracy") is None
        assert template_for("accuracy") is None

    def test_the_5f_admission_is_refused_by_name(self) -> None:
        """Attention-head observables parse but are not in the table; the
        refusal names every unmeasurable string and is the typed
        refusal the experimentation stage recognises."""
        predictions = tuple(admitted(metric) for metric in FIVE_F_METRICS)

        with pytest.raises(UnmeasurablePredictionsError) as caught:
            require_measurable(predictions)

        message = str(caught.value)
        for metric in FIVE_F_METRICS:
            assert metric in message
        assert isinstance(caught.value, ExperimentationUnavailableError)
        assert isinstance(
            caught.value, ExperimentationStage().refusals()
        )

    def test_a_state_with_no_predictions_is_refused(self) -> None:
        with pytest.raises(UnmeasurablePredictionsError, match="nothing"):
            require_measurable(())


class TestTheCatalog:
    def test_substitution_reaches_every_placeholder(self) -> None:
        catalog = catalog_for((admitted(ENCODER_METRIC),), trainer="stub")

        (entry,) = catalog.entries
        assert PLACEHOLDER not in entry.template.source
        assert ENCODER_METRIC in entry.template.source
        assert entry.metrics[0] == ENCODER_METRIC
        assert entry.control is not None
        assert entry.control.metric in entry.metrics
        compile(entry.template.source, entry.template.name, "exec")

    def test_the_real_templates_build_too(self) -> None:
        catalog = catalog_for((admitted(ENCODER_METRIC),), trainer="real")
        (entry,) = catalog.entries
        assert "torchvision" in entry.template.source
        compile(entry.template.source, entry.template.name, "exec")

    def test_gpu_occupancy_reaches_the_estimate(self) -> None:
        catalog = catalog_for(
            (admitted(ENCODER_METRIC),), trainer="real", gpu_count=1
        )
        (entry,) = catalog.entries
        assert entry.estimated_cost.gpu_hours == pytest.approx(
            entry.estimated_cost.wall_clock_seconds / 3600.0
        )

    def test_fences_partition_and_fill_slot_preserves_them(self) -> None:
        catalog = catalog_for((admitted(ENCODER_METRIC),), trainer="stub")
        (entry,) = catalog.entries
        source = entry.template.source

        regions = fixed_regions(source)
        completed = fill_slot(source, STUB_SLOT)

        assert len(regions) >= 2
        assert fixed_regions(completed) == regions
        assert "NotImplementedError" not in completed
        compile(completed, "completed", "exec")

    def test_an_unservable_trainer_refuses(self) -> None:
        augment = (
            "difference in linear probe accuracy: "
            "augmented training minus plain training"
        )
        with pytest.raises(UnmeasurablePredictionsError, match="stub"):
            catalog_for((admitted(augment),), trainer="stub")


class TestThePlannerGateFit:
    """Task 7B prep: the per-run catalog is something ``check_decision``
    can hold a planner to."""

    def payload(
        self, catalog: TemplateCatalog, metric: str
    ) -> dict[str, object]:
        (entry,) = catalog.entries
        return {
            "action": "new_experiment",
            "rationale": "test the admitted contrast",
            "question_id": "q_1",
            "evidence_ids": ["ev_1"],
            "hypothesis_id": "hyp_1",
            "hypothesis_statement": "",
            "prediction_condition": "on held-out images",
            "prediction_metric": metric,
            "prediction_comparator": "gt",
            "prediction_threshold": 0.0,
            "prediction_tolerance": 0,
            "prediction_expectation": "the contrast is positive",
            "experiment_objective": "measure the contrast",
            "experiment_procedure": "run the catalog template",
            "experiment_metrics": [metric],
            "experiment_baselines": ["the untrained arm"],
            "experiment_controls": [],
            "experiment_seeds": [11],
            "template_id": entry.template.id,
            "target_experiment_id": "",
            "replication_seed": -1,
            "removed_component": "",
            "stop_reason": "none",
        }

    def rules(self, catalog: TemplateCatalog, metric: str) -> set[str]:

        question = ResearchQuestion(text="What does training explain?")
        context = RoleContext(
            questions=(question,),
            hypotheses=(
                Hypothesis(
                    statement="Training helps.", question_id=question.id
                ),
            ),
        )
        payload = self.payload(catalog, metric)
        payload["question_id"] = context.questions[0].id
        payload["hypothesis_id"] = context.hypotheses[0].id
        rejections = check_decision(
            payload, context=context, catalog=catalog
        )
        return {r.rule for r in rejections}

    def test_the_admitted_metric_is_declarable(self) -> None:
        catalog = catalog_for((admitted(ENCODER_METRIC),), trainer="stub")
        assert "undeclared_metric" not in self.rules(catalog, ENCODER_METRIC)
        assert "unknown_template" not in self.rules(catalog, ENCODER_METRIC)

    def test_a_foreign_metric_is_rejected(self) -> None:
        catalog = catalog_for((admitted(ENCODER_METRIC),), trainer="stub")
        assert "undeclared_metric" in self.rules(catalog, "made_up_top1")


class TestFundedStateDiscovery:
    def request(self, tmp_path: Path) -> RuntimeRequest:
        return RuntimeRequest(
            root=tmp_path,
            evidence=FileEvidenceStore(tmp_path),
            states=FileStateStore(tmp_path),
            ledger=BudgetLedger(tmp_path / "program", "run_1"),
            journal=RunJournal(tmp_path / "program", "run_1"),
            bundles=CommitBundleStore(tmp_path / "program"),
        )

    def test_no_funded_run_is_a_typed_refusal(self, tmp_path: Path) -> None:
        with pytest.raises(
            ExperimentationUnavailableError, match="no funded run"
        ):
            _funded_state(self.request(tmp_path))


class TestTheWholeWalk:
    """The CI qualification: seven stages, real subprocesses, zero
    network, zero torch, zero docker."""

    def test_the_stub_chain_walks_to_an_honest_end(
        self, tmp_path: Path
    ) -> None:
        result = walk(tmp_path / "run", lab=ci_lab())

        assert result.ok
        assert str(result.outcome) == "ended"
        assert "no open scientific work remains" in result.detail

        report = verify_run(
            tmp_path / "run", program_root=tmp_path / "run" / "program"
        )
        assert report.ok, report.issues

        # The science that walk recorded: every seeded run verified,
        # every sign test consistent, the claim assessed SUPPORTED.
        states = FileStateStore(tmp_path / "run")
        loaded = [states.load(found) for found in states.state_ids()]
        head = max(loaded, key=lambda state: len(state.attempts))
        assert len(head.prediction_tests) == 3
        assert all(
            str(test.consistency) == "consistent"
            for test in head.prediction_tests
        )
        assert any(
            str(found.verdict) == "supported" for found in head.assessments
        )
        verifications = tmp_path / "run" / "verifications"
        records = [
            json.loads(path.read_text())
            for path in verifications.glob("*.json")
        ]
        assert len(records) == 3
        assert all(
            record["standing"] == "verified_evidence" for record in records
        )

    def test_the_walk_survives_a_stop_and_a_resume(
        self, tmp_path: Path
    ) -> None:
        from autonomous_research_lab.control.stage import StageName

        first = walk(
            tmp_path / "run", lab=ci_lab(), stop_after=StageName.FUNDING
        )
        assert str(first.outcome) == "stopped"

        second = walk(tmp_path / "run", lab=ci_lab())

        assert second.ok
        report = verify_run(
            tmp_path / "run", program_root=tmp_path / "run" / "program"
        )
        assert report.ok, report.issues
