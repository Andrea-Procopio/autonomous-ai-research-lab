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
   execution, and exactly one other package depends on it. A selection
   is a preference among survivors, never a scientific proposition.
9. ``admission`` is that consumer — the governed bridge from one
   SELECTED selection into the initial ``ResearchState``. It reads
   ``core`` (including, uniquely among these stages, the state it
   constructs), ``ideation``, ``mapping``, ``priorart``, ``selection``,
   ``persistence``, and the provider seam — never ``literature``. It
   builds the genesis state in one constructor call, never calls a
   state mutator, and nothing in the package depends on it. The
   analysis chain is a DAG with one invariant: retrieved papers and
   everything derived from them have no path into scientific state
   except through admission's governed door.
10. ``program`` is admission's one consumer — the funded run an
   admitted state descends into, and the one place that can say whether
   a whole run is intact. It reads ``core``, ``admission``,
   ``persistence``, and ``evidence``, and nothing else; of every state
   mutator it may call only ``fund``, because a funded successor is the
   last state this side of the loop builds and every evolution after it
   belongs to orchestration's commit layer. It cannot reach roles,
   orchestration, execution, or literature, and nothing imports it.
11. ``evidence`` stays a foundation, not a consumer: results, evidence
   records, their artifacts, and the derived chain checker depend on
   ``core`` alone. Verifying a run needs snapshots and a ledger too,
   which is exactly why that pass lives one layer up in ``program``
   rather than here.
12. ``control`` is the composition root: the one package that may import
   every stage, and the one nothing may import. It sequences stages and
   records what happened to them; it mutates no state and builds no
   result or evidence record of its own. Rules 4 through 10 name it as
   an allowed consumer for exactly this reason — a composition root is
   the thing they forbid *everywhere else*.
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
PROGRAM = SRC / "program"
EVIDENCE = SRC / "evidence"
CONTROL = SRC / "control"
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
        "fund",
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


def test_selection_has_exactly_one_consumer_in_the_package() -> None:
    """Only ``admission`` — the governed bridge into the initial
    research state — may import selection: a selection record is a
    validated model preference, and admission is the one task that may
    act on it."""
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


def test_priorart_is_imported_only_by_its_deliberate_stages() -> None:
    """Only ``selection`` — the choice among the challenge's survivors —
    and ``admission`` — which re-verifies the challenged portfolio
    behind its own door — may import priorart. A prior-art assessment
    reaches scientific state only through that chain."""
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


def test_admission_depends_only_on_its_declared_inputs() -> None:
    """The admitter reads ``core`` (including, uniquely, the state it
    constructs), ``ideation`` (the candidate), ``mapping`` (the shared
    gate vocabulary), ``priorart`` and ``selection`` (the lineage it
    verifies), ``persistence`` (the snapshot store), and the provider
    seam — pointedly not ``literature``: admission runs no retrieval.
    An admission module that needed roles, orchestration, evidence, or
    execution would be doing science, not translation."""
    allowed_absolute = (
        f"{PACKAGE}.core",
        f"{PACKAGE}.ideation",
        f"{PACKAGE}.mapping",
        f"{PACKAGE}.priorart",
        f"{PACKAGE}.selection",
        f"{PACKAGE}.admission",
        f"{PACKAGE}.persistence",
        f"{PACKAGE}.runtime.providers",
        f"{PACKAGE}.runtime.metrics",
    )
    allowed_relative = (
        "core",
        "ideation",
        "mapping",
        "priorart",
        "selection",
        "persistence",
        "runtime.providers",
        "runtime.metrics",
    )
    violations: list[str] = []
    for path in sorted(ADMISSION.rglob("*.py")):
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


def test_admission_constructs_the_state_but_never_mutates_one() -> None:
    """Admission is the one analysis-side package allowed to construct
    a ``ResearchState`` — and only to construct it: the genesis state is
    built in a single constructor call, and no admission module may call
    a state mutator. Evolution after genesis belongs to orchestration's
    commit layer, exactly as it does for roles."""
    violations: list[str] = []
    for path in sorted(ADMISSION.rglob("*.py")):
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


def test_admission_cannot_reach_evidence_or_execution() -> None:
    """The construct-only exception is narrow: admission may reach the
    state it builds, but never the proposal, transition, evidence, or
    execution machinery, and never ``Evidence`` or ``ExperimentResult``
    by name — a result can only come from a process that ran."""
    forbidden_names = {"Evidence", "ExperimentResult"}
    violations: list[str] = []
    for path in sorted(ADMISSION.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module
                    for fragment in (
                        "proposals",
                        "transitions",
                        "evidence",
                        "execution",
                    )
                ):
                    violations.append(f"{path.name}: imports {module}")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    f"{path.name}:{node.lineno}: references {node.id}"
                )
    assert violations == []


def test_admission_imports_no_vendor_sdk() -> None:
    """The admitter is provider-neutral: it speaks to the generic seam,
    and no Muse-specific (or any vendor-specific) logic may appear
    here."""
    for path in sorted(ADMISSION.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"
        modules = set(_imported_modules(path))
        assert not any("muse" in module for module in modules), (
            f"{path.name} imports the Muse adapter; the admitter knows "
            f"only the generic provider seam"
        )


def test_admission_has_exactly_one_consumer_in_the_package() -> None:
    """Only ``program`` — the funded run an admitted state descends
    into — may import admission: an admission record is a translated
    seed, and funding it into a run is the one act that follows."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if ADMISSION in path.parents or PROGRAM in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "admission" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "admission" in alias.name.split(".")
                )
    assert violations == []


def test_program_depends_only_on_its_declared_inputs() -> None:
    """The run bridge reads ``core`` (the state it funds), ``admission``
    (the seed it funds), ``persistence`` (the snapshot store), and
    ``evidence`` (the facts a run accumulates, for the integrity pass) —
    and nothing else. It has no provider seam because it makes no model
    call: funding is an operator act and a deterministic one. A program
    module that needed roles, orchestration, literature, or the analysis
    stages behind admission would be re-deciding what admission already
    decided."""
    allowed_absolute = (
        f"{PACKAGE}.core",
        f"{PACKAGE}.admission",
        f"{PACKAGE}.evidence",
        f"{PACKAGE}.persistence",
        f"{PACKAGE}.program",
    )
    allowed_relative = ("core", "admission", "evidence", "persistence")
    violations: list[str] = []
    for path in sorted(PROGRAM.rglob("*.py")):
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


def test_program_calls_only_the_funding_mutator() -> None:
    """Funding a genesis state is this package's one act on a state.
    Every other evolution — proposals, attempts, results, judgments —
    belongs to orchestration's commit layer, exactly as it does for
    roles, so no other mutator may be called here."""
    violations: list[str] = []
    for path in sorted(PROGRAM.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in STATE_MUTATORS
                and node.func.attr != "fund"
            ):
                violations.append(
                    f"{path.name}:{node.lineno}: calls .{node.func.attr}(...)"
                )
    assert violations == []


def test_program_cannot_reach_evidence_or_execution() -> None:
    """A grant buys the chance to find something out; it never records
    what was found. The integrity pass reads the evidence stores, so the
    package may import them — but no module here may import the
    proposal, transition, or execution machinery, and none may name
    ``Evidence`` or ``ExperimentResult`` as a type it constructs."""
    forbidden_names = {"Evidence", "ExperimentResult"}
    violations: list[str] = []
    for path in sorted(PROGRAM.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    fragment in module.split(".")
                    for fragment in (
                        "proposals",
                        "transitions",
                        "execution",
                        "roles",
                        "orchestration",
                        "literature",
                    )
                ):
                    violations.append(f"{path.name}: imports {module}")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(
                    f"{path.name}:{node.lineno}: references {node.id}"
                )
    assert violations == []


def test_program_imports_no_vendor_sdk() -> None:
    """Funding makes no model call, so no adapter — vendor or house —
    has any business here."""
    for path in sorted(PROGRAM.rglob("*.py")):
        roots = {module.split(".")[0] for module in _imported_modules(path)}
        assert roots & VENDOR_SDKS == set(), f"{path.name} imports a vendor SDK"
        modules = set(_imported_modules(path))
        assert not any("muse" in module for module in modules), (
            f"{path.name} imports the Muse adapter; funding makes no call"
        )


def test_nothing_in_the_package_imports_program() -> None:
    """Program is the new leaf. The funded state leaves it the way the
    admitted state left admission: through the snapshot store and an
    explicit hand-off at the edge, never through an import of the
    package by anything upstream of it."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if PROGRAM in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "program" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "program" in alias.name.split(".")
                )
    assert violations == []


def test_evidence_depends_on_core_alone() -> None:
    """Results, evidence records, their artifacts, and the chain checker
    are the foundation every later layer rests on, so they rest on
    nothing but the scientific vocabulary. Verifying a whole run needs
    snapshots and a budget ledger as well, which is exactly why that
    pass lives in ``program`` and not here."""
    violations: list[str] = []
    for path in sorted(EVIDENCE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"{PACKAGE}."):
                    if not module.startswith(f"{PACKAGE}.core"):
                        violations.append(f"{path.name}: imports {module}")
                elif node.level == 2 and not module.startswith("core"):
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
                    and not alias.name.startswith(f"{PACKAGE}.core")
                )
    assert violations == []


def test_nothing_in_the_package_imports_control() -> None:
    """The controller is the composition root and therefore the leaf.
    It is allowed to import every stage precisely because nothing
    imports it: the arrow that would let a stage learn its own position
    in the chain is the one this rule refuses to draw."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if CONTROL in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "control" in module.split("."):
                    violations.append(
                        f"{path.relative_to(SRC)}: imports {module}"
                    )
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.relative_to(SRC)}: imports {alias.name}"
                    for alias in node.names
                    if "control" in alias.name.split(".")
                )
    assert violations == []


def test_control_sequences_stages_but_never_does_their_work() -> None:
    """A composition root that started deciding things would be a second
    implementation of every stage it calls. The controller may hold a
    state and hand it on; it may not mutate one, and it may not build a
    result or an evidence record — those belong to the runtime and the
    stage that owns them."""
    forbidden = {"ExperimentResult", "Evidence", "EpistemicAssessment"}
    violations: list[str] = []
    for path in sorted(CONTROL.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in STATE_MUTATORS
                ):
                    violations.append(
                        f"{path.name}:{node.lineno}: calls "
                        f".{node.func.attr}(...)"
                    )
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in forbidden
                ):
                    violations.append(
                        f"{path.name}:{node.lineno}: constructs "
                        f"{node.func.id}"
                    )
    assert violations == []
