"""Structural invariants, enforced rather than documented.

1. ``core`` imports nothing from its sibling packages: the scientific
   vocabulary must stay independent of the machinery around it.
2. ``roles`` never mutates ``ResearchState``: roles produce proposals, and
   only the transition layer commits. A role calling a state mutator would
   bypass validation, provenance, and any future conflict resolution.
3. the model-provider boundary stays provider-neutral and stays away from
   scientific state: no vendor SDK import, no ``ResearchState``.
4. ``literature`` is a leaf boundary: it depends on ``core`` alone, cannot
   reach scientific state, evidence, or execution, and exactly one other
   package depends on it — literature describes what external papers
   report, and only a later, separate stage may decide what that means.
5. ``mapping`` is that stage, and it is bounded the same way: it may read
   ``core``, ``literature``, and the provider seam, and nothing else; it
   cannot reach scientific state, evidence, or execution, and exactly one
   other package depends on it — literature analysis proposes maps, and
   only the deliberate idea-generation stage may decide what they
   suggest.
6. ``ideation`` is that stage, bounded the same way again: it may read
   ``core``, ``mapping``, and the provider seam — never ``literature``
   directly; sources reach candidates only as opaque ids on mapping
   records — it cannot reach scientific state, evidence, or execution,
   and exactly one other package depends on it. Candidate ideas are
   conjectures carrying their sources, never propositions.
7. ``priorart`` is one such consumer — the adversarial challenge that
   tries to falsify each candidate's differentiation against fresh
   bounded retrieval. It reads ``core``, ``literature`` (it runs its
   own searches), ``mapping`` (the grounding surface and shared gate
   vocabulary), ``ideation`` (the portfolio it challenges), and the
   provider seam; it cannot reach scientific state, evidence, or
   execution, and exactly one other package depends on it — a challenge
   verdict is an input to deliberate selection, never to anything else.
8. ``selection`` is that consumer — the choice among the challenge's
   survivors. It reads ``core``, ``ideation``, ``mapping``,
   ``priorart``, and the provider seam — never ``literature``: it runs
   no retrieval and sees sources only through the records upstream
   stages froze. It cannot reach scientific state, evidence, or
   execution, and nothing in the package depends on it. A selection is
   a preference among survivors, never a scientific proposition. The
   analysis chain is a DAG with one invariant: retrieved papers and
   everything derived from them have no path into scientific state.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "autonomous_research_lab"
SRC = Path(__file__).resolve().parents[1] / "src" / PACKAGE
CORE = SRC / "core"
ROLES = SRC / "roles"
LITERATURE = SRC / "literature"
MAPPING = SRC / "mapping"
IDEATION = SRC / "ideation"
PRIORART = SRC / "priorart"
SELECTION = SRC / "selection"
ADMISSION = SRC / "admission"
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


def test_literature_is_imported_only_by_its_analysis_stages() -> None:
    """Only ``mapping`` and ``priorart`` — the deliberate
    literature-analysis stages, both barred from scientific state — may
    import literature. No orchestration, role, runtime, or evidence
    module may depend on it: retrieved papers must have no path into
    scientific state."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if (
            LITERATURE in path.parents
            or MAPPING in path.parents
            or PRIORART in path.parents
        ):
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


def test_mapping_depends_only_on_its_declared_inputs() -> None:
    """The field mapper reads ``core``, ``literature``, and the provider
    seam (``runtime.providers`` / ``runtime.metrics``) — nothing else in
    the package. A mapping module that needed roles, orchestration,
    evidence, or execution would be doing science, not analysis."""
    allowed_absolute = (
        f"{PACKAGE}.core",
        f"{PACKAGE}.literature",
        f"{PACKAGE}.mapping",
        f"{PACKAGE}.runtime.providers",
        f"{PACKAGE}.runtime.metrics",
    )
    allowed_relative = ("core", "literature", "runtime.providers", "runtime.metrics")
    violations: list[str] = []
    for path in sorted(MAPPING.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"{PACKAGE}."):
                    if not module.startswith(allowed_absolute):
                        violations.append(f"{path.name}: imports {module}")
                elif node.level == 2 and not module.startswith(
                    allowed_relative
                ):
                    violations.append(
                        f"{path.name}: relative import of {module}"
                    )
                elif node.level > 2:
                    violations.append(
                        f"{path.name}: relative import above package"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE}.")
                    and not alias.name.startswith(allowed_absolute)
                )
    assert violations == []


def test_mapping_cannot_touch_scientific_state() -> None:
    """Literature analysis creates no scientific-state propositions: no
    module in ``mapping`` may import the state, proposal, transition,
    evidence, or execution machinery, reference ``ResearchState``,
    ``Evidence``, or ``ExperimentResult`` by name, or call a state
    mutator. A field map is a description of the literature, never a
    fact of this lab's."""
    forbidden_names = {"ResearchState", "Evidence", "ExperimentResult"}
    violations: list[str] = []
    for path in sorted(MAPPING.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module
                    for fragment in (
                        ".state",
                        "proposals",
                        "transitions",
                        "evidence",
                        "execution",
                    )
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
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    f"{path.name}:{node.lineno}: references {node.id}"
                )
    assert violations == []


def test_mapping_imports_no_vendor_sdk() -> None:
    """The mapper is provider-neutral: it speaks to the generic seam, and
    no Muse-specific (or any vendor-specific) logic may appear here."""
    for path in sorted(MAPPING.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"
        modules = set(_imported_modules(path))
        assert not any("muse" in module for module in modules), (
            f"{path.name} imports the Muse adapter; the mapper knows only "
            f"the generic provider seam"
        )


def test_mapping_is_imported_only_by_its_analysis_stages() -> None:
    """Only ``ideation`` — the deliberate idea-generation stage Task 5C
    built — ``priorart`` — the challenge stage that shares its gate
    vocabulary and grounding surface — ``selection`` — the choice
    stage that reuses the same vocabulary — and ``admission`` — the
    governed bridge that reuses it again — may import mapping. No
    orchestration, role, runtime, or evidence module may depend on it:
    a field map must have no path into scientific state."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if (
            MAPPING in path.parents
            or IDEATION in path.parents
            or PRIORART in path.parents
            or SELECTION in path.parents
            or ADMISSION in path.parents
        ):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "mapping" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "mapping" in alias.name.split(".")
                )
    assert violations == []


def test_ideation_depends_only_on_its_declared_inputs() -> None:
    """The idea generator reads ``core``, ``mapping``, and the provider
    seam (``runtime.providers`` / ``runtime.metrics``) — nothing else in
    the package, and pointedly not ``literature``: sources reach
    candidates only as the opaque ids mapping records carry. An ideation
    module that needed roles, orchestration, evidence, or execution
    would be doing science, not conjecture."""
    allowed_absolute = (
        f"{PACKAGE}.core",
        f"{PACKAGE}.mapping",
        f"{PACKAGE}.ideation",
        f"{PACKAGE}.runtime.providers",
        f"{PACKAGE}.runtime.metrics",
    )
    allowed_relative = ("core", "mapping", "runtime.providers", "runtime.metrics")
    violations: list[str] = []
    for path in sorted(IDEATION.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"{PACKAGE}."):
                    if not module.startswith(allowed_absolute):
                        violations.append(f"{path.name}: imports {module}")
                elif node.level == 2 and not module.startswith(
                    allowed_relative
                ):
                    violations.append(
                        f"{path.name}: relative import of {module}"
                    )
                elif node.level > 2:
                    violations.append(
                        f"{path.name}: relative import above package"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE}.")
                    and not alias.name.startswith(allowed_absolute)
                )
    assert violations == []


def test_ideation_cannot_touch_scientific_state() -> None:
    """A candidate idea is a conjecture, never a scientific-state
    proposition: no module in ``ideation`` may import the state,
    proposal, transition, evidence, or execution machinery, reference
    ``ResearchState``, ``Evidence``, or ``ExperimentResult`` by name, or
    call a state mutator. Admission through the governed commit belongs
    to a later task."""
    forbidden_names = {"ResearchState", "Evidence", "ExperimentResult"}
    violations: list[str] = []
    for path in sorted(IDEATION.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module
                    for fragment in (
                        ".state",
                        "proposals",
                        "transitions",
                        "evidence",
                        "execution",
                    )
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
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    f"{path.name}:{node.lineno}: references {node.id}"
                )
    assert violations == []


def test_ideation_imports_no_vendor_sdk() -> None:
    """The generator is provider-neutral: it speaks to the generic seam,
    and no Muse-specific (or any vendor-specific) logic may appear
    here."""
    for path in sorted(IDEATION.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"
        modules = set(_imported_modules(path))
        assert not any("muse" in module for module in modules), (
            f"{path.name} imports the Muse adapter; the generator knows "
            f"only the generic provider seam"
        )


def test_ideation_is_imported_only_by_its_deliberate_stages() -> None:
    """Only ``priorart`` — the challenge stage that reads the portfolio
    it tries to falsify — and ``selection`` — the choice among the
    challenge's survivors — may import ideation, both barred from
    scientific state: candidate ideas reach scientific state only when
    a later task deliberately builds the proposal path through the
    governed commit."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if (
            IDEATION in path.parents
            or PRIORART in path.parents
            or SELECTION in path.parents
            or ADMISSION in path.parents
        ):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "ideation" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "ideation" in alias.name.split(".")
                )
    assert violations == []


def test_priorart_depends_only_on_its_declared_inputs() -> None:
    """The prior-art challenger reads ``core``, ``literature`` (it runs
    its own bounded searches), ``mapping`` (the shared gate vocabulary
    and grounding surface), ``ideation`` (the portfolio it challenges),
    and the provider seam (``runtime.providers`` / ``runtime.metrics``)
    — nothing else in the package. A priorart module that needed roles,
    orchestration, evidence, or execution would be doing science, not
    challenge."""
    allowed_absolute = (
        f"{PACKAGE}.core",
        f"{PACKAGE}.literature",
        f"{PACKAGE}.mapping",
        f"{PACKAGE}.ideation",
        f"{PACKAGE}.priorart",
        f"{PACKAGE}.runtime.providers",
        f"{PACKAGE}.runtime.metrics",
    )
    allowed_relative = (
        "core",
        "literature",
        "mapping",
        "ideation",
        "runtime.providers",
        "runtime.metrics",
    )
    violations: list[str] = []
    for path in sorted(PRIORART.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"{PACKAGE}."):
                    if not module.startswith(allowed_absolute):
                        violations.append(f"{path.name}: imports {module}")
                elif node.level == 2 and not module.startswith(
                    allowed_relative
                ):
                    violations.append(
                        f"{path.name}: relative import of {module}"
                    )
                elif node.level > 2:
                    violations.append(
                        f"{path.name}: relative import above package"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE}.")
                    and not alias.name.startswith(allowed_absolute)
                )
    assert violations == []


def test_priorart_cannot_touch_scientific_state() -> None:
    """A prior-art verdict is a judgment about a bounded corpus, never a
    scientific-state proposition: no module in ``priorart`` may import
    the state, proposal, transition, evidence, or execution machinery,
    reference ``ResearchState``, ``Evidence``, or ``ExperimentResult``
    by name, or call a state mutator. Admission through the governed
    commit belongs to a later task."""
    forbidden_names = {"ResearchState", "Evidence", "ExperimentResult"}
    violations: list[str] = []
    for path in sorted(PRIORART.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module
                    for fragment in (
                        ".state",
                        "proposals",
                        "transitions",
                        "evidence",
                        "execution",
                    )
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
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    f"{path.name}:{node.lineno}: references {node.id}"
                )
    assert violations == []


def test_priorart_imports_no_vendor_sdk() -> None:
    """The challenger is provider-neutral: it speaks to the generic
    seam, and no Muse-specific (or any vendor-specific) logic may
    appear here."""
    for path in sorted(PRIORART.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"
        modules = set(_imported_modules(path))
        assert not any("muse" in module for module in modules), (
            f"{path.name} imports the Muse adapter; the challenger knows "
            f"only the generic provider seam"
        )


def test_selection_depends_only_on_its_declared_inputs() -> None:
    """The selector reads ``core``, ``ideation`` (the portfolio),
    ``mapping`` (the shared gate vocabulary), ``priorart`` (the
    verdicts that define eligibility), and the provider seam
    (``runtime.providers`` / ``runtime.metrics``) — pointedly not
    ``literature``: selection runs no retrieval and sees sources only
    through the records upstream stages froze. A selection module that
    needed roles, orchestration, evidence, or execution would be doing
    science, not choice."""
    allowed_absolute = (
        f"{PACKAGE}.core",
        f"{PACKAGE}.ideation",
        f"{PACKAGE}.mapping",
        f"{PACKAGE}.priorart",
        f"{PACKAGE}.selection",
        f"{PACKAGE}.runtime.providers",
        f"{PACKAGE}.runtime.metrics",
    )
    allowed_relative = (
        "core",
        "ideation",
        "mapping",
        "priorart",
        "runtime.providers",
        "runtime.metrics",
    )
    violations: list[str] = []
    for path in sorted(SELECTION.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"{PACKAGE}."):
                    if not module.startswith(allowed_absolute):
                        violations.append(f"{path.name}: imports {module}")
                elif node.level == 2 and not module.startswith(
                    allowed_relative
                ):
                    violations.append(
                        f"{path.name}: relative import of {module}"
                    )
                elif node.level > 2:
                    violations.append(
                        f"{path.name}: relative import above package"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE}.")
                    and not alias.name.startswith(allowed_absolute)
                )
    assert violations == []


def test_selection_cannot_touch_scientific_state() -> None:
    """A selection is a preference over a challenged portfolio, never a
    scientific-state proposition: no module in ``selection`` may import
    the state, proposal, transition, evidence, or execution machinery,
    reference ``ResearchState``, ``Evidence``, or ``ExperimentResult``
    by name, or call a state mutator. Admission through the governed
    commit belongs to a later task."""
    forbidden_names = {"ResearchState", "Evidence", "ExperimentResult"}
    violations: list[str] = []
    for path in sorted(SELECTION.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module
                    for fragment in (
                        ".state",
                        "proposals",
                        "transitions",
                        "evidence",
                        "execution",
                    )
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
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    f"{path.name}:{node.lineno}: references {node.id}"
                )
    assert violations == []


def test_selection_imports_no_vendor_sdk() -> None:
    """The selector is provider-neutral: it speaks to the generic seam,
    and no Muse-specific (or any vendor-specific) logic may appear
    here."""
    for path in sorted(SELECTION.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"
        modules = set(_imported_modules(path))
        assert not any("muse" in module for module in modules), (
            f"{path.name} imports the Muse adapter; the selector knows "
            f"only the generic provider seam"
        )


def test_nothing_in_the_package_imports_selection() -> None:
    """Selection is the new leaf: a selected candidate reaches
    scientific state only when a later task deliberately builds the
    proposal path through the governed commit."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if SELECTION in path.parents or ADMISSION in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "selection" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "selection" in alias.name.split(".")
                )
    assert violations == []


def test_priorart_has_exactly_one_consumer_in_the_package() -> None:
    """Only ``selection`` — the choice among the challenge's survivors,
    itself barred from scientific state — may import priorart: a
    prior-art assessment reaches scientific state only when a later
    task deliberately builds the proposal path through the governed
    commit."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if (
            PRIORART in path.parents
            or SELECTION in path.parents
            or ADMISSION in path.parents
        ):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "priorart" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "priorart" in alias.name.split(".")
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
