"""The model-backed research engineer: spec in, executed result out.

The first concrete :class:`~autonomous_research_lab.roles.base.ResearchRole`
with a model behind it, for the ``RESEARCH_ENGINEER`` seat. One invocation
performs one narrow slice of work::

    ExperimentSpec (from the invocation context, never from state)
      -> one structured model call        (schema-validated locally)
      -> deterministic source validation  (allowlist, size, text, syntax)
      -> preserved, content-addressed implementation + provenance record
      -> ExperimentJob                    (constructed by trusted code only)
      -> the existing Executor            (which alone produces results)
      -> exactly one ResultProposal

The model's authority is deliberately narrow. It may propose the *content*
of exactly one allowlisted Python file (``experiment.py``) and the rationale
behind it — nothing else. It chooses no commands, no paths, no dependencies,
no environment variables, and it can state no metrics: every number the lab
sees comes from the ``metrics.json`` the executed process wrote, read by the
executor. A reply that strays — an unlisted or unsafe path, oversized or
non-text content, code that does not compile — is rejected deterministically
*before* anything executes, and the rejected attempt is preserved as data.

Provider failures are runtime failures: the observed accounting is folded
into the :class:`~autonomous_research_lab.runtime.providers.UsageLedger`
exactly once and the typed error is re-raised — no result, no evidence, no
scientific residue. And a completed run with disappointing metrics is not a
failure of any kind here: this role returns what actually happened and holds
no opinion about it.

The role depends only on the provider-neutral seam
(:class:`~autonomous_research_lab.runtime.providers.ModelProvider`); which
vendor sits behind it is wiring. How validated source is *launched* is a
:class:`~autonomous_research_lab.execution.binding.JobBinding` — live
model-generated code belongs in the container binding, never directly on
the host.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, NoReturn

from ..core.actions import ResearchAction, ResearchActionType
from ..core.experiment import ExperimentSpec
from ..core.ids import content_id
from ..core.proposals import Proposal, ResultProposal
from ..core.state import ResearchState
from ..execution.binding import JobBinding
from ..execution.executor import Executor
from ..runtime.implementation_store import (
    ImplementationRecord,
    ImplementationStore,
    SourceFile,
    sha256_text,
)
from ..runtime.preflight import (
    DEFAULT_PREFLIGHT_CHECKS,
    PreflightCheck,
    require_preflight,
)
from ..runtime.providers import (
    Message,
    MessageRole,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OutputSchema,
    UsageLedger,
)
from .base import ResearchRole, RoleInvocation, RoleName, RoleSuitability

ENTRYPOINT: Final = "experiment.py"
ALLOWED_SOURCE_FILES: Final = frozenset({ENTRYPOINT})
MAX_SOURCE_BYTES: Final = 128 * 1024

_POLL_SECONDS: Final = 0.05

#: The whole output contract of the model call. ``files`` is constrained
#: further by deterministic validation (exactly the allowlisted set); the
#: schema deliberately has nowhere to put a metric, a result, or a claim.
PROPOSAL_SCHEMA: Final = OutputSchema(
    name="implementation_proposal",
    json_schema={
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "description": (
                    "The complete source files of the implementation."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Relative file name; must be exactly "
                                "'experiment.py'."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "The full file content.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            "rationale": {
                "type": "string",
                "description": (
                    "How the implementation realizes the specification: "
                    "data generation, algorithm, how each declared metric "
                    "is computed, and how determinism is ensured."
                ),
            },
        },
        "required": ["files", "rationale"],
        "additionalProperties": False,
    },
)

ENGINEER_INSTRUCTION: Final = (
    "You are the research engineer of an autonomous scientific laboratory. "
    "You implement experiment specifications as small, deterministic, "
    "standard-library-only Python programs. You return exactly the JSON the "
    "schema requires: the source files and a short implementation "
    "rationale. You never report metrics, results, or conclusions; the "
    "program you write is what measures them. Use only plain ASCII "
    "characters in the source you produce."
)


class EngineerContractError(RuntimeError):
    """The invocation itself is unusable — an unsupported action, or a
    context without exactly one experiment spec. Deterministic, raised
    before any model call or execution; no spend, no side effects."""


class ImplementationRejectedError(RuntimeError):
    """The model's reply failed deterministic source validation and was
    refused before any execution. The attempt (payload and reason) is
    preserved in the implementation store, never silently discarded."""


@dataclass(frozen=True, slots=True)
class ImplementationTemplate:
    """The fixed starting point the model completes — identified content,
    so the record can say exactly which template a generation began from."""

    name: str
    source: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.source)

    @property
    def id(self) -> str:
        return content_id("tmpl", self.name, self.sha256)


class ModelBackedEngineer(ResearchRole):
    """See the module docstring; construction is explicit wiring, and every
    collaborator is injected — provider, executor, ledger, store, binding,
    template — so tests and live runs differ only in what is plugged in."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        executor: Executor,
        ledger: UsageLedger,
        store: ImplementationStore,
        binding: JobBinding,
        template: ImplementationTemplate,
        template_resolver: (
            Callable[[ExperimentSpec], ImplementationTemplate | None] | None
        ) = None,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        request_timeout_seconds: float = 240.0,
        max_generation_repairs: int = 1,
        preflight_checks: Sequence[PreflightCheck] = DEFAULT_PREFLIGHT_CHECKS,
    ) -> None:
        self._provider = provider
        self._model = model
        self._executor = executor
        self._ledger = ledger
        self._store = store
        self._binding = binding
        self._template = template
        self._template_resolver = template_resolver
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._request_timeout_seconds = request_timeout_seconds
        self._max_generation_repairs = max_generation_repairs
        self._preflight_checks = tuple(preflight_checks)

    @property
    def name(self) -> RoleName:
        return RoleName.RESEARCH_ENGINEER

    @property
    def supported_actions(self) -> frozenset[ResearchActionType]:
        return frozenset(
            {ResearchActionType.RUN_EXPERIMENT, ResearchActionType.REPLICATE}
        )

    def suitability(
        self,
        state: ResearchState,  # noqa: ARG002 - static seat, static fit
        action: ResearchAction,
    ) -> RoleSuitability:
        fits = action.action_type in self.supported_actions
        return RoleSuitability(
            value=1.0 if fits else 0.0,
            rationale="the one model-backed executor seat",
        )

    def perform(self, invocation: RoleInvocation) -> tuple[Proposal, ...]:
        action_type = invocation.assignment.action_type
        if action_type not in self.supported_actions:
            raise EngineerContractError(
                f"the engineer performs run_experiment and replicate, "
                f"not {action_type}"
            )
        spec = _the_spec(invocation)
        seed = _seed_for(spec, invocation)
        template = self._template_for(spec)

        request = self._request(invocation, spec, seed, template)
        response = self._invoke(request)
        # Bounded generation repair, the smallest model-backed response to
        # the one failure class observed live (reply text corrupted before
        # it ever became source): a deterministically rejected reply earns
        # at most ``max_generation_repairs`` corrective calls, each carrying
        # the exact rejection reason. Every rejected attempt has already
        # been preserved by the time the retry happens; a repair that fails
        # too re-raises the rejection. Pre-execution only — repairing a run
        # that *executed* stays the debug loop's job, not this role's.
        repairs = 0
        while True:
            try:
                files, rationale = self._validated_source(
                    invocation, spec, response
                )
                break
            except ImplementationRejectedError as rejection:
                if repairs >= self._max_generation_repairs:
                    raise
                repairs += 1
                request = _repair_request(
                    request, response, str(rejection), repairs
                )
                response = self._invoke(request)

        source_id, source_dir = self._store.persist_source(files)
        implementation_id = self._implementation_id(
            invocation,
            spec,
            source_id,
            files,
            seed,
            request,
            response,
            rationale,
            template,
        )
        job = self._binding.bind(
            spec_id=spec.id,
            source_dir=source_dir,
            entrypoint=ENTRYPOINT,
            config={
                "spec_id": spec.id,
                "source_id": source_id,
                "implementation_id": implementation_id,
            },
            seed=seed,
        )
        record = ImplementationRecord(
            invocation_id=invocation.id,
            action_type=action_type.value,
            spec_id=spec.id,
            template_id=template.id,
            template_sha256=template.sha256,
            source_id=source_id,
            manifest={f.path: f.sha256 for f in files},
            entrypoint=ENTRYPOINT,
            command=job.command,
            config=job.config,
            seed=seed,
            required_artifacts=job.required_artifacts,
            request_fingerprint=request.fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=self._model,
            served_model=response.model,
            provider_request_id=response.request_id,
            rationale=rationale,
        )
        assert record.id == implementation_id  # same identity fields, twice
        self._store.record(record)

        # The record now exists whatever happens next: a preflight
        # rejection or a failed run preserves, never erases, the attempt.
        require_preflight(job, spec, checks=self._preflight_checks)
        job_id = self._executor.submit(job)
        while not self._executor.status(job_id).is_terminal:
            time.sleep(_POLL_SECONDS)
        result = self._executor.collect(job_id)
        return (
            ResultProposal(
                result=result,
                proposer=f"engineer:{self._provider.name}:{self._model}",
            ),
        )

    # -- the model call ------------------------------------------------------

    def _template_for(self, spec: ExperimentSpec) -> ImplementationTemplate:
        """The template this spec starts from: the resolver's answer when
        one is wired and it knows the spec (planner-selected templates
        reach the engineer this way), else the constructor's template."""
        if self._template_resolver is not None:
            resolved = self._template_resolver(spec)
            if resolved is not None:
                return resolved
        return self._template

    def _request(
        self,
        invocation: RoleInvocation,
        spec: ExperimentSpec,
        seed: int | None,
        template: ImplementationTemplate,
    ) -> ModelRequest:
        return ModelRequest(
            model=self._model,
            instruction=ENGINEER_INSTRUCTION,
            messages=(
                Message(
                    role=MessageRole.USER,
                    content=_render_task(spec, seed, template),
                ),
            ),
            schema=PROPOSAL_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            timeout_seconds=self._request_timeout_seconds,
            metadata={
                "invocation_id": invocation.id,
                "spec_id": spec.id,
                "action": invocation.assignment.action_type.value,
            },
        )

    def _invoke(self, request: ModelRequest) -> ModelResponse:
        """One provider call, with its accounting always reaching the
        ledger: the response on success, the attached cost on failure —
        recorded exactly once — before the typed error propagates."""
        try:
            response = self._provider.invoke(request)
        except ModelProviderError as error:
            self._ledger.record_failure(error)
            raise
        self._ledger.record(response)
        return response

    # -- deterministic source validation -------------------------------------

    def _validated_source(
        self,
        invocation: RoleInvocation,
        spec: ExperimentSpec,
        response: ModelResponse,
    ) -> tuple[tuple[SourceFile, ...], str]:
        payload = response.structured
        if payload is None:
            self._reject(
                invocation, spec, response, "the reply carried no structured payload"
            )
        try:
            files, rationale = _parse_proposal(payload)
        except ImplementationRejectedError as error:
            self._reject(invocation, spec, response, str(error), payload=payload)
        return files, rationale

    def _reject(
        self,
        invocation: RoleInvocation,
        spec: ExperimentSpec,
        response: ModelResponse,
        reason: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> NoReturn:
        self._store.preserve_rejected(
            invocation_id=invocation.id,
            spec_id=spec.id,
            reason=reason,
            request_fingerprint=response.request_fingerprint,
            response_id=response.id,
            payload=payload if payload is not None else response.text,
        )
        raise ImplementationRejectedError(
            f"implementation rejected before execution: {reason}"
        )

    def _implementation_id(
        self,
        invocation: RoleInvocation,
        spec: ExperimentSpec,
        source_id: str,
        files: tuple[SourceFile, ...],
        seed: int | None,
        request: ModelRequest,
        response: ModelResponse,
        rationale: str,
        template: ImplementationTemplate,
    ) -> str:
        """The record id, needed *before* the job exists so the job's config
        can carry it. Identity excludes binding output (command, config,
        required artifacts), so this provisional record derives the same id
        as the final one — asserted at the call site."""
        return ImplementationRecord(
            invocation_id=invocation.id,
            action_type=invocation.assignment.action_type.value,
            spec_id=spec.id,
            template_id=template.id,
            template_sha256=template.sha256,
            source_id=source_id,
            manifest={f.path: f.sha256 for f in files},
            entrypoint=ENTRYPOINT,
            command=(),
            config={},
            seed=seed,
            required_artifacts=(),
            request_fingerprint=request.fingerprint,
            response_id=response.id,
            provider=response.provider,
            requested_model=self._model,
            served_model=response.model,
            provider_request_id=response.request_id,
            rationale=rationale,
        ).id


def _repair_request(
    base: ModelRequest,
    failed: ModelResponse,
    reason: str,
    attempt: int,
) -> ModelRequest:
    """One corrective request: the failed reply plus the exact deterministic
    rejection reason. The ASCII reminder answers the failure class observed
    live — non-ASCII characters corrupted into unusable bytes inside the
    reply — without loosening any validation."""
    feedback = (
        f"{reason}. Nothing was executed. Return the complete corrected "
        f"files now, satisfying every original constraint, and use only "
        f"plain ASCII characters (printable ASCII, spaces and newlines) "
        f"in the source."
    )
    return ModelRequest(
        model=base.model,
        instruction=base.instruction,
        messages=(
            *base.messages,
            Message(
                role=MessageRole.ASSISTANT,
                content=failed.text or "(empty reply)",
            ),
            Message(role=MessageRole.USER, content=feedback),
        ),
        schema=base.schema,
        max_output_tokens=base.max_output_tokens,
        temperature=base.temperature,
        timeout_seconds=base.timeout_seconds,
        metadata={**base.metadata, "generation_repair": str(attempt)},
    )


def _the_spec(invocation: RoleInvocation) -> ExperimentSpec:
    specs = invocation.context.experiments
    if len(specs) != 1:
        raise EngineerContractError(
            f"the engineer needs exactly one experiment spec in its "
            f"context, got {len(specs)}"
        )
    return specs[0]


def _seed_for(spec: ExperimentSpec, invocation: RoleInvocation) -> int | None:
    """The first declared seed no prior run of this spec has used; when
    every declared seed has run (a replication), the first declared seed —
    an exact rerun is a legitimate replication. ``None`` only when the spec
    declares no seeds."""
    if not spec.seeds:
        return None
    used = {r.seed for r in invocation.context.results if r.spec_id == spec.id}
    return next((s for s in spec.seeds if s not in used), spec.seeds[0])


def _parse_proposal(
    payload: Mapping[str, object],
) -> tuple[tuple[SourceFile, ...], str]:
    """Deterministic validation of a schema-valid proposal, fail-closed.

    The schema already guarantees shape and types; this enforces what a
    schema cannot: the file-set allowlist, path safety, content that is
    real UTF-8 text of bounded size, and source that compiles. Raises
    :class:`ImplementationRejectedError` with the first offence."""
    raw_files = payload["files"]
    rationale = payload["rationale"]
    assert isinstance(raw_files, Sequence)
    assert isinstance(rationale, str)
    if not raw_files:
        raise ImplementationRejectedError("the reply proposes no files")

    files: list[SourceFile] = []
    seen: set[str] = set()
    for entry in raw_files:
        assert isinstance(entry, Mapping)
        path = entry["path"]
        content = entry["content"]
        assert isinstance(path, str)
        assert isinstance(content, str)
        _check_path(path, seen)
        _check_content(path, content)
        seen.add(path)
        files.append(SourceFile(path=path, content=content))
    if ENTRYPOINT not in seen:
        raise ImplementationRejectedError(
            f"the entrypoint {ENTRYPOINT!r} is missing from the proposed files"
        )
    return tuple(files), rationale


def _check_path(path: str, seen: set[str]) -> None:
    pure = PurePosixPath(path)
    if not path.strip():
        raise ImplementationRejectedError("a proposed file has an empty path")
    if pure.is_absolute() or path.startswith(("/", "\\")) or ":" in path:
        raise ImplementationRejectedError(
            f"absolute path {path!r} is not allowed"
        )
    if ".." in pure.parts or "\\" in path:
        raise ImplementationRejectedError(
            f"path {path!r} attempts traversal outside the workspace"
        )
    if path in seen:
        raise ImplementationRejectedError(f"duplicate file path {path!r}")
    if path not in ALLOWED_SOURCE_FILES:
        raise ImplementationRejectedError(
            f"file {path!r} is outside the allowlist "
            f"({', '.join(sorted(ALLOWED_SOURCE_FILES))})"
        )


def _check_content(path: str, content: str) -> None:
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ImplementationRejectedError(
            f"{path} is not valid text: {exc}"
        ) from exc
    if "\x00" in content:
        raise ImplementationRejectedError(f"{path} contains a NUL byte")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ImplementationRejectedError(
            f"{path} is {len(encoded)} bytes, over the "
            f"{MAX_SOURCE_BYTES}-byte limit"
        )
    try:
        compile(content, path, "exec")
    except (SyntaxError, ValueError) as exc:
        raise ImplementationRejectedError(
            f"{path} does not compile: {exc}"
        ) from exc


def _render_task(
    spec: ExperimentSpec, seed: int | None, template: ImplementationTemplate
) -> str:
    metrics = ", ".join(spec.metrics)
    baselines = "; ".join(spec.baselines) if spec.baselines else "none declared"
    controls = "; ".join(spec.controls) if spec.controls else "none declared"
    seed_line = (
        f"{seed} (arrives as the environment variable ARL_SEED)"
        if seed is not None
        else "none declared; the program must still be deterministic"
    )
    return f"""Implement the experiment specification below as exactly one \
Python file.

## Specification
- objective: {spec.objective}
- procedure: {spec.procedure}
- declared metrics (every one must appear in metrics.json): {metrics}
- baselines: {baselines}
- controls: {controls}
- seed for this run: {seed_line}

## Execution contract
- The program runs unattended, with no arguments and no user input.
- Read the environment variables ARL_RUN_DIR (the output directory), \
ARL_CONFIG (the path to a JSON config file), and ARL_SEED (an integer as \
text; seed all randomness from it).
- Write ARL_RUN_DIR/metrics.json: one flat JSON object mapping every \
declared metric name to a finite number. Extra numeric metrics are \
allowed; a missing or non-finite declared metric fails the run.
- Write any other output files only inside ARL_RUN_DIR.
- Exit with status 0 on success.

## Implementation constraints
- Exactly one file, named experiment.py. No other files.
- Python 3.11, standard library only. No third-party packages, no pip.
- No network access; the execution environment has none.
- Do not read or write anything outside ARL_RUN_DIR (reading the \
ARL_CONFIG file is fine).
- No subprocesses and no threads.
- Deterministic: the same ARL_SEED must reproduce the same metrics.
- Keep the file well under {MAX_SOURCE_BYTES // 1024} KiB.

## Template
Start from this template (identifier {template.id}, sha256 \
{template.sha256}); keep its structure and complete it into a full \
implementation of the procedure above:

```python
{template.source}
```"""
