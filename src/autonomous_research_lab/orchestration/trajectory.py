"""Trajectory logging: preserving decisions for later scientific evaluation.

Every orchestration decision is a data point of the form

    (state_t, {candidate_i, utility_i}, selected_t, outcome_t, state_{t+1})

and questions this project intends to answer — is the utility estimate
calibrated? do specialized roles help? does the search policy earn its cost? —
are all questions *about these tuples*. So they are preserved from the first
run, in full, failures included.

The logger is deliberately primitive: one JSON object per line, appended to a
local file. No database, no schema migration, no indexing — a trajectory is
read by analysis code, and JSONL is enough until real volume says otherwise.
Timestamps are stamped here, at the infrastructure edge, keeping ``core``
free of wall-clock reads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..core.decision import DecisionRecord


class JsonlTrajectoryLogger:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: DecisionRecord) -> None:
        payload = record.to_jsonable()
        assert isinstance(payload, dict)
        payload["logged_at"] = datetime.now(tz=UTC).isoformat()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def read(self) -> tuple[dict[str, object], ...]:
        """Parse the log back as plain dictionaries, for analysis and tests."""
        if not self._path.exists():
            return ()
        with self._path.open(encoding="utf-8") as handle:
            return tuple(json.loads(line) for line in handle if line.strip())
