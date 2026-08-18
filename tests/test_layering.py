"""Structural invariants, enforced rather than documented.

1. ``core`` imports nothing from its sibling packages: the scientific
   vocabulary must stay independent of the machinery around it.
2. ``roles`` never mutates ``ResearchState``: roles produce proposals, and
   only the transition layer commits. A role calling a state mutator would
   bypass validation, provenance, and any future conflict resolution.
3. the model-provider boundary stays provider-neutral and stays away from
   scientific state: no vendor SDK import, no ``ResearchState``.
4. ``literature`` is a leaf boundary: it depends on ``core`` alone, cannot
   reach scientific state, evidence, or execution, and nothing else in the
   package depends on it — literature describes what external papers
   report, and only a later, separate stage may decide what that means.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "autonomous_research_lab"
SRC = Path(__file__).resolve().parents[1] / "src" / PACKAGE
CORE = SRC / "core"
ROLES = SRC / "roles"
LITERATURE = SRC / "literature"
PROVIDERS = SRC / "runtime" / "providers.py"
PROVIDER_MODULES = (PROVIDERS, SRC / "runtime" / "muse.py")

#: Vendor SDKs that must never be imported by the provider-neutral seam. An
#: adapter for one of these belongs in its own module, behind the interface.
VENDOR_SDKS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "cohere",
        "mistralai",
        "ollama",
        "litellm",
        "httpx",
        "requests",
        "urllib3",
    }
)

#: Every state-mutating method on ResearchState (plus its private seam).
STATE_MUTATORS = frozenset(
    {
        "upsert_question",
        "upsert_hypothesis",
        "upsert_prediction",
        "add_experiment",
        "record_result",
        "record_evidence",
        "record_prediction_test",
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


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def test_the_provider_seam_imports_no_vendor_sdk() -> None:
    """The interface is provider-neutral by construction: a concrete adapter
    imports its SDK inside its own module, never here."""
    roots = {module.split(".")[0] for module in _imported_modules(PROVIDERS)}
    assert roots & VENDOR_SDKS == set()


def test_the_provider_seam_cannot_touch_scientific_state() -> None:
    """Model output is text until a role turns it into a proposal and the
    transition layer commits it. Neither the seam nor an adapter has a way
    to shortcut that — provider failures stay runtime events."""
    violations: list[str] = []
    for path in PROVIDER_MODULES:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "state" in module or "proposals" in module:
                    violations.append(f"{path.name}: imports {module}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in STATE_MUTATORS
            ):
                violations.append(
                    f"{path.name}:{node.lineno}: calls .{node.func.attr}(...)"
                )
            elif isinstance(node, ast.Name) and node.id == "ResearchState":
                violations.append(
                    f"{path.name}:{node.lineno}: references ResearchState"
                )
    assert violations == []


def test_roles_never_construct_experiment_results() -> None:
    """Results come from executors that ran processes. A role that could
    construct an ``ExperimentResult`` could synthesise a fact from
    reasoning, which is the one proposal payload roles must never build."""
    violations: list[str] = []
    for path in sorted(ROLES.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name == "ExperimentResult":
                violations.append(
                    f"{path.name}:{node.lineno}: constructs ExperimentResult"
                )
    assert violations == []


def test_literature_depends_on_core_alone() -> None:
    """The literature boundary sees the identity and type helpers in
    ``core`` and nothing else in the package: no runtime, no roles, no
    orchestration, no evidence, no execution. A literature module that
    needed any of those would be doing science, not retrieval."""
    allowed = (f"{PACKAGE}.core", f"{PACKAGE}.literature")
    violations: list[str] = []
    for path in sorted(LITERATURE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"{PACKAGE}."):
                    if not module.startswith(allowed):
                        violations.append(f"{path.name}: imports {module}")
                elif node.level == 2 and not module.startswith("core"):
                    violations.append(
                        f"{path.name}: relative import of {module}"
                    )
                elif node.level > 2:
                    violations.append(f"{path.name}: relative import above package")
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE}.")
                    and not alias.name.startswith(allowed)
                )
    assert violations == []


def test_literature_cannot_touch_scientific_state() -> None:
    """A retrieved paper is a description of external work, never a
    scientific fact of this lab's: no module in ``literature`` may name
    ``ResearchState``, call a state mutator, or import the state,
    proposal, or transition machinery."""
    violations: list[str] = []
    for path in sorted(LITERATURE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module
                    for fragment in ("state", "proposals", "transitions")
                ):
                    violations.append(f"{path.name}: imports {module}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in STATE_MUTATORS
            ):
                violations.append(
                    f"{path.name}:{node.lineno}: calls .{node.func.attr}(...)"
                )
            elif isinstance(node, ast.Name) and node.id == "ResearchState":
                violations.append(
                    f"{path.name}:{node.lineno}: references ResearchState"
                )
    assert violations == []


def test_literature_imports_no_vendor_sdk() -> None:
    """The retrieval seam keeps the zero-dependency promise the model seam
    keeps: stdlib HTTP only, no requests/httpx/urllib3 and kin."""
    for path in sorted(LITERATURE.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"


def test_nothing_else_in_the_package_imports_literature() -> None:
    """Literature is a leaf: until a later task deliberately wires field
    mapping into the loop, no orchestration, role, runtime, or evidence
    module may depend on it — retrieved papers must have no path into
    scientific state."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if LITERATURE in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "literature" in module:
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "literature" in alias.name
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
