"""The implementation store: an id never maps to different source.

Same invariant family as the evidence and verification stores — write-once,
idempotent for identical content, loud on conflict and corruption — applied
to the one artifact class those stores do not cover: the generated source
that actually ran, and the generation event that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autonomous_research_lab.runtime.implementation_store import (
    ImplementationConflictError,
    ImplementationIntegrityError,
    ImplementationRecord,
    ImplementationStore,
    SourceFile,
    source_tree_id,
)

FILES = (SourceFile(path="experiment.py", content="print('hello')\n"),)


def _record(
    *,
    response_id: str = "mcall_0000000000000001",
    command: tuple[str, ...] = ("python", "experiment.py"),
    rationale: str = "a fixture implementation",
) -> ImplementationRecord:
    source_id = source_tree_id(FILES)
    return ImplementationRecord(
        invocation_id="inv_0000000000000001",
        action_type="run_experiment",
        spec_id="exp_0000000000000001",
        template_id="tmpl_0000000000000001",
        template_sha256="ab" * 32,
        source_id=source_id,
        manifest={f.path: f.sha256 for f in FILES},
        entrypoint="experiment.py",
        command=command,
        config={"spec_id": "exp_0000000000000001", "seedless": False},
        seed=7,
        required_artifacts=("metrics.json",),
        request_fingerprint="mreq_0000000000000001",
        response_id=response_id,
        provider="fake",
        requested_model="test-model",
        served_model="test-model-v2",
        provider_request_id="req-42",
        rationale=rationale,
    )


def test_source_trees_are_content_addressed(tmp_path: Path) -> None:
    store = ImplementationStore(tmp_path)
    source_id, tree = store.persist_source(FILES)
    assert source_id == source_tree_id(FILES)
    assert (tree / "experiment.py").read_text() == FILES[0].content
    # Idempotent: the same content re-persists as a no-op, same id and dir.
    again_id, again_tree = store.persist_source(FILES)
    assert (again_id, again_tree) == (source_id, tree)


def test_a_tampered_source_tree_is_a_loud_conflict(tmp_path: Path) -> None:
    store = ImplementationStore(tmp_path)
    source_id, tree = store.persist_source(FILES)
    (tree / "experiment.py").write_text("print('tampered')\n")
    with pytest.raises(ImplementationConflictError):
        store.persist_source(FILES)
    assert source_id  # the name survives; its content mismatch is the error


def test_records_round_trip_and_are_idempotent(tmp_path: Path) -> None:
    store = ImplementationStore(tmp_path)
    record = _record()
    stored = store.record(record)
    assert stored == record
    assert store.record(record) == record  # identical re-record: no-op
    loaded = store.get(record.id)
    assert loaded == record
    assert store.records() == (record,)


def test_same_id_cannot_name_a_different_binding(tmp_path: Path) -> None:
    """Command/config are excluded from the id derivation (they embed the id
    itself), so the write-once guard is what pins them: re-recording the
    same generation with a different command must raise, not overwrite."""
    store = ImplementationStore(tmp_path)
    record = _record()
    variant = _record(command=("python", "-O", "experiment.py"))
    assert record.id == variant.id
    store.record(record)
    with pytest.raises(ImplementationConflictError):
        store.record(variant)
    assert store.get(record.id) == record


def test_distinct_generation_events_are_distinct_records(
    tmp_path: Path,
) -> None:
    store = ImplementationStore(tmp_path)
    first = _record(response_id="mcall_0000000000000001")
    second = _record(response_id="mcall_0000000000000002")
    assert first.source_id == second.source_id  # identical source ...
    assert first.id != second.id  # ... two events, two records
    store.record(first)
    store.record(second)
    assert len(store.records()) == 2


def test_a_hand_edited_record_fails_on_load(tmp_path: Path) -> None:
    store = ImplementationStore(tmp_path)
    record = store.record(_record())
    path = tmp_path / "implementations" / f"{record.id}.json"
    payload = json.loads(path.read_text())
    payload["rationale"] = "quietly rewritten"
    path.write_text(json.dumps(payload))
    with pytest.raises(ImplementationIntegrityError):
        store.get(record.id)


def test_rejected_attempts_are_preserved_as_data(tmp_path: Path) -> None:
    store = ImplementationStore(tmp_path)
    written = store.preserve_rejected(
        invocation_id="inv_0000000000000001",
        spec_id="exp_0000000000000001",
        reason="file '/etc/passwd' is outside the allowlist",
        request_fingerprint="mreq_0000000000000001",
        response_id="mcall_0000000000000001",
        payload={"files": [{"path": "/etc/passwd", "content": "x"}]},
    )
    assert written.exists()
    (entry,) = store.rejected()
    assert entry["reason"] == "file '/etc/passwd' is outside the allowlist"
    payload = entry["payload"]
    assert isinstance(payload, dict)
    # Preserved as JSON, never materialized as a file at the unsafe path.
    assert not (tmp_path / "etc").exists()
    assert not Path("/etc/passwd").read_text().startswith("x")
