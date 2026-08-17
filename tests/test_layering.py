"""Structural invariants, enforced rather than documented.

1. ``core`` imports nothing from its sibling packages: the scientific
   vocabulary must stay independent of the machinery around it.
2. ``roles`` never mutates ``ResearchState``: roles produce proposals, and
   only the transition layer commits. A role calling a state mutator would
   bypass validation, provenance, and any future conflict resolution.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "autonomous_research_lab"
SRC = Path(__file__).resolve().parents[1] / "src" / PACKAGE
CORE = SRC / "core"
ROLES = SRC / "roles"

#: Every state-mutating method on ResearchState (plus its private seam).
STATE_MUTATORS = frozenset(
    {
        "upsert_question",
        "upsert_hypothesis",
        "upsert_prediction",
        "add_experiment",
        "record_result",
        "record_evidence",
        "upsert_claim",
        "link_evidence",
        "record_assessment",
        "begin_attempt",
        "resolve_attempt",
        "apply",
        "charge",
        "_evolve",
    }
)


def _is_foreign(module: str | None) -> bool:
    """True for an import of a sibling package inside ``autonomous_research_lab``."""
    if module is None:
        return False
    return module.startswith(f"{PACKAGE}.") and not module.startswith(f"{PACKAGE}.core")


def test_core_depends_on_nothing_else_in_the_package() -> None:
    violations: list[str] = []
    for path in sorted(CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level >= 2:
                    violations.append(f"{path.name}: relative import above core")
                elif _is_foreign(node.module):
                    violations.append(f"{path.name}: imports {node.module}")
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if _is_foreign(alias.name)
                )
    assert violations == []


def test_roles_never_call_state_mutators() -> None:
    """AST-level enforcement of the proposal invariant: no module in
    ``roles/`` calls any ResearchState mutator. (Method-name matching is
    deliberately blunt — a false positive here is a prompt to move the code,
    not to widen the rule.)"""
    violations: list[str] = []
    for path in sorted(ROLES.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in STATE_MUTATORS
            ):
                violations.append(
                    f"{path.name}:{node.lineno}: calls .{node.func.attr}(...)"
                )
    assert violations == []


def test_roles_do_not_import_the_transition_layer() -> None:
    """Roles do not commit their own proposals either — committing is the
    orchestrator's job, or the separation is theatre."""
    violations: list[str] = []
    for path in sorted(ROLES.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "transitions" in module or (
                    node.level >= 2 and "orchestration" in module
                ):
                    violations.append(f"{path.name}: imports {module}")
    assert violations == []
