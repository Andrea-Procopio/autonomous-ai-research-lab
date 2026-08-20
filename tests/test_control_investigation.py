"""The investigation record and the store that keeps it.

The claim under test is that a run's parameters are provenance: the
config is stored verbatim, addressed by its own content, named by the
investigation, and loud if anyone edits it afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.control.investigation import (
    MAX_LABEL_CHARS,
    Investigation,
    InvestigationConflictError,
    InvestigationIntegrityError,
    InvestigationStore,
)
from autonomous_research_lab.control.stage import StageName

CONFIG: dict[str, object] = {
    "label": "toy chain",
    "brief": {"topic": "sample efficiency", "cutoff_date": "2026-01-01"},
    "grant": {"usd": 10.0},
}


def investigation(**overrides: object) -> Investigation:
    fields: dict[str, object] = {
        "investigation_id": "inv_1",
        "config_id": "cfg_1",
        "label": "toy chain",
    }
    fields.update(overrides)
    return Investigation(**fields)  # type: ignore[arg-type]


class TestTheRecord:
    def test_it_must_name_its_config(self) -> None:
        with pytest.raises(ValueError, match="config_id"):
            investigation(config_id="  ")

    def test_a_label_is_a_handle_not_a_description(self) -> None:
        with pytest.raises(ValueError, match="at most"):
            investigation(label="x" * (MAX_LABEL_CHARS + 1))

    def test_stop_after_must_name_a_stage(self) -> None:
        with pytest.raises(ValueError, match="names no stage"):
            investigation(stop_after="publication")

    def test_stop_after_accepts_a_stage(self) -> None:
        assert investigation(stop_after=str(StageName.FUNDING)).stop_after == (
            "funding"
        )

    def test_identity_covers_the_content(self) -> None:
        assert investigation().id != investigation(label="other").id
        assert investigation().id == investigation().id


class TestTheStore:
    def test_a_config_is_addressed_by_its_own_content(
        self, tmp_path: Path
    ) -> None:
        store = InvestigationStore(tmp_path)

        first = store.record_config(CONFIG)
        second = store.record_config(dict(CONFIG))

        assert first == second
        assert first.startswith("cfg_")

    def test_a_config_round_trips_verbatim(self, tmp_path: Path) -> None:
        store = InvestigationStore(tmp_path)
        config_id = store.record_config(CONFIG)

        assert InvestigationStore(tmp_path).get_config(config_id) == CONFIG

    def test_an_edited_config_is_loud(self, tmp_path: Path) -> None:
        store = InvestigationStore(tmp_path)
        config_id = store.record_config(CONFIG)
        path = tmp_path / "configs" / f"{config_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["grant"]["usd"] = 10_000.0
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(InvestigationIntegrityError, match="was edited"):
            store.get_config(config_id)

    def test_an_unknown_config_is_absent_not_an_error(
        self, tmp_path: Path
    ) -> None:
        assert InvestigationStore(tmp_path).get_config("cfg_nothing") is None

    def test_an_investigation_round_trips(self, tmp_path: Path) -> None:
        store = InvestigationStore(tmp_path)
        store.record(investigation())

        found = InvestigationStore(tmp_path).get("inv_1")

        assert found == investigation()

    def test_recording_the_same_investigation_twice_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        store = InvestigationStore(tmp_path)
        store.record(investigation())
        store.record(investigation())

        assert len(store.investigations()) == 1

    def test_one_investigation_id_is_stated_once(self, tmp_path: Path) -> None:
        store = InvestigationStore(tmp_path)
        store.record(investigation())

        with pytest.raises(InvestigationConflictError, match="stated once"):
            store.record(investigation(label="a different label"))

    def test_two_investigations_can_share_one_config(
        self, tmp_path: Path
    ) -> None:
        store = InvestigationStore(tmp_path)
        store.record(investigation(investigation_id="inv_1"))
        store.record(investigation(investigation_id="inv_2"))

        assert len(store.investigations()) == 2
        assert len(list((tmp_path / "investigations").glob("*.json"))) == 2

    def test_a_tampered_investigation_is_loud(self, tmp_path: Path) -> None:
        store = InvestigationStore(tmp_path)
        record = store.record(investigation())
        path = tmp_path / "investigations" / f"{record.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["label"] = "something else entirely"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with pytest.raises(InvestigationIntegrityError, match="re-derives"):
            store.investigations()

    def test_the_log_belongs_to_the_investigation(self, tmp_path: Path) -> None:
        store = InvestigationStore(tmp_path)
        store.record(investigation())

        log = store.log_for("inv_1")

        assert log.investigation_id == "inv_1"
        assert log.directory == tmp_path / "logs" / "inv_1"
