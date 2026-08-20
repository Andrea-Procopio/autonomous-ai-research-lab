"""One investigation's parameters, read from one file.

The claim under test is that a config which cannot produce a legal run
is refused now — before a store is touched, a provider is built, or a
single call is made — and that the document the operator wrote is what
gets recorded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from autonomous_research_lab.control.config import (
    ConfigError,
    load_config,
    parse_config,
)
from autonomous_research_lab.control.stage import StageName

CONFIG: dict[str, Any] = {
    "label": "toy chain",
    "model": "fake-model-1",
    "brief": {
        "topic": "sample efficiency in tiny classifiers",
        "cutoff_date": "2026-01-01",
        "recent_window_start": "2024-01-01",
        "max_queries_per_family": 1,
        "results_per_query": 4,
        "max_screened_sources": 8,
        "max_extracted_sources": 4,
        "max_model_calls": 12,
        "refinement_rounds": 0,
    },
    "cfp": {
        "source_url": "https://example.invalid/cfp",
        "supplied_at": "2026-02-01T00:00:00+00:00",
        "text": "A workshop on sample efficiency. Submissions due soon.",
    },
    "ideation": {"max_candidates": 2, "max_model_calls": 2},
    "selection": {
        "compute_constraint": "One CPU, no accelerator.",
        "data_constraint": "Synthetic data only.",
        "time_constraint": "Under ten minutes per run.",
        "experimental_constraint": "Deterministic seeds.",
        "max_eligible_candidates": 2,
    },
    "admission": {
        "scheduling_requirement": "Run jobs one at a time.",
        "job_duration_requirement": "No job over ten minutes.",
        "checkpoint_requirement": "No checkpoints needed.",
    },
    "funding": {
        "granted": {
            "wall_clock_seconds": 3600.0,
            "usd": 5.0,
            "model_tokens": 200_000,
        },
        "authority": "Lab operator, for the canary run.",
    },
}


def config(**overrides: object) -> dict[str, Any]:
    merged: dict[str, Any] = json.loads(json.dumps(CONFIG))
    merged.update(overrides)
    return merged


def without(section: str, key: str) -> dict[str, Any]:
    merged = config()
    del merged[section][key]
    return merged


def with_in(section: str, key: str, value: object) -> dict[str, Any]:
    merged = config()
    merged[section][key] = value
    return merged


class TestAGoodConfig:
    def test_it_parses_into_typed_settings(self) -> None:
        parsed = parse_config(CONFIG)

        assert parsed.label == "toy chain"
        assert parsed.brief.topic.startswith("sample efficiency")
        assert parsed.ideation.max_candidates == 2
        assert parsed.funding.granted.usd == 5.0
        assert parsed.experimentation.max_steps == 24

    def test_the_challenge_window_defaults_to_the_briefs(self) -> None:
        parsed = parse_config(CONFIG)

        assert parsed.prior_art.cutoff_date == "2026-01-01"
        assert parsed.prior_art.recent_window_start == "2024-01-01"

    def test_a_stated_challenge_window_overrides_it(self) -> None:
        parsed = parse_config(
            config(prior_art={"cutoff_date": "2025-06-01",
                              "recent_window_start": "2023-01-01"})
        )

        assert parsed.prior_art.cutoff_date == "2025-06-01"

    def test_the_run_label_defaults_to_the_investigations(self) -> None:
        assert parse_config(CONFIG).funding.label == "toy chain"

    def test_the_snapshot_hashes_its_own_text(self) -> None:
        assert len(parse_config(CONFIG).snapshot.text_sha256) == 64

    def test_stop_after_names_a_stage(self) -> None:
        parsed = parse_config(config(stop_after="funding"))

        assert parsed.stop_after is StageName.FUNDING

    def test_no_stop_after_means_the_whole_chain(self) -> None:
        assert parse_config(CONFIG).stop_after is None

    def test_it_holds_no_ids(self) -> None:
        """The one thing a config may never do is name a record, which is
        the hand-bridging this package exists to remove."""
        flat = json.dumps(CONFIG)
        for prefix in ("madq_", "irun_", "prun_", "srun_", "arun_", "rune_"):
            assert prefix not in flat


class TestWhatItRefuses:
    def test_a_missing_section(self) -> None:
        merged = config()
        del merged["funding"]

        with pytest.raises(ConfigError, match="missing required section"):
            parse_config(merged)

    def test_a_missing_key(self) -> None:
        with pytest.raises(ConfigError, match="missing required key 'topic'"):
            parse_config(without("brief", "topic"))

    def test_an_unknown_key_is_a_typo_not_a_default(self) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            parse_config(with_in("brief", "max_quereis_per_family", 3))

    def test_an_unknown_top_level_key(self) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            parse_config(config(templates=["mine.py"]))

    def test_a_date_that_is_not_a_date(self) -> None:
        with pytest.raises(ConfigError, match=r"brief: .*ISO date"):
            parse_config(with_in("brief", "cutoff_date", "January 2026"))

    def test_a_window_that_ends_before_it_starts(self) -> None:
        with pytest.raises(ConfigError, match="brief:"):
            parse_config(with_in("brief", "recent_window_start", "2027-01-01"))

    def test_a_cap_above_its_stages_ceiling(self) -> None:
        with pytest.raises(ConfigError, match="illegal run"):
            parse_config(with_in("ideation", "max_candidates", 99))

    def test_a_constraint_longer_than_its_stage_allows(self) -> None:
        with pytest.raises(ConfigError, match="illegal run"):
            parse_config(with_in("selection", "data_constraint", "x" * 500))

    def test_a_grant_above_the_authorized_ceiling(self) -> None:
        merged = config()
        merged["funding"]["granted"]["usd"] = 25_000.0

        with pytest.raises(ConfigError, match="above the authorized ceiling"):
            parse_config(merged)

    def test_a_grant_that_buys_nothing(self) -> None:
        merged = config()
        merged["funding"]["granted"] = {"usd": 0.0}

        with pytest.raises(ConfigError, match="buys nothing"):
            parse_config(merged)

    def test_an_authority_longer_than_the_record_allows(self) -> None:
        with pytest.raises(ConfigError, match=r"funding\.authority"):
            parse_config(with_in("funding", "authority", "x" * 500))

    def test_a_step_count_outside_its_range(self) -> None:
        with pytest.raises(ConfigError, match="max_steps"):
            parse_config(config(experimentation={"max_steps": 0}))

    def test_a_stop_after_that_names_nothing(self) -> None:
        with pytest.raises(ConfigError, match="names no stage"):
            parse_config(config(stop_after="publication"))

    def test_a_number_where_a_string_belongs(self) -> None:
        with pytest.raises(ConfigError, match="must be a string"):
            parse_config(with_in("brief", "topic", 7))

    def test_a_string_where_a_number_belongs(self) -> None:
        with pytest.raises(ConfigError, match="must be an integer"):
            parse_config(with_in("ideation", "max_candidates", "two"))


class TestLoadingAFile:
    def test_it_returns_the_document_it_read(self, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

        parsed, payload = load_config(path)

        assert parsed.label == "toy chain"
        assert payload == CONFIG

    def test_the_document_is_the_operators_own(self, tmp_path: Path) -> None:
        """Verbatim, not re-serialized from the parsed settings: what was
        asked for is the record, and what the codec understood is
        derived."""
        path = tmp_path / "run.json"
        document: Mapping[str, object] = config(stop_after="admission")
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")

        _, payload = load_config(path)

        assert payload["stop_after"] == "admission"

    def test_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot read"):
            load_config(tmp_path / "nothing.json")

    def test_a_file_that_is_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        path.write_text("label: toy chain\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(path)

    def test_a_file_that_is_not_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ConfigError, match="must hold a JSON object"):
            load_config(path)
