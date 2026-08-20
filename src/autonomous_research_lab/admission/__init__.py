"""Governed admission: one validated selection becomes the initial state.

The analysis chain ends here::

    literature -> mapping -> ideation -> priorart -> selection -> admission

An :class:`AdmissionDirective` names one selection run record — never a
"latest" anything. Trusted code verifies the complete lineage behind it
(the selection run, its directive, the prior-art run, the ideation run,
the direction, the CFP snapshot, every candidate and assessment),
requires the outcome ``SELECTED``, and retrieves exactly the selected
candidate. One gated model call — at most one corrective — translates
that candidate's recorded predictions into machine-checkable core
predictions under the neutral sign-only encoding; everything else the
initial state needs is a deterministic verbatim copy. The admitted
:class:`~..core.state.ResearchState` is built in one constructor call,
persisted content-addressed, and recorded beside a write-once
:class:`AdmissionRecord` — all or nothing, one admission per selection
run, ever, with a completed directive replaying its stored result at
zero model calls.

The ladder, stated once: ``DISTINGUISHED`` meant differentiated within
one bounded prior-art corpus; ``SELECTED`` meant preferred within one
constrained portfolio; ``ADMITTED`` means converted into the governed
research state. None of the three means true, novel, or empirically
supported.

Binding negatives: no result, evidence, assessment, claim, or novelty
status is expressible anywhere in this vocabulary; the admitted state
holds propositions only, with a zero budget (budget assignment is later
operator work); the candidate records, the selection record, and every
other upstream artifact stay byte-identical; and admission never calls
a state mutator — evolution after the genesis state belongs to the
orchestration commit layer.
"""

from ..mapping.gates import MappingRejection
from .admitter import (
    OPERATIONALIZATION_INSTRUCTION,
    AdmissionBudgetError,
    AdmissionRejectedError,
    AdmissionRunResult,
    CandidateAdmitter,
    build_field_texts,
    encoded_metric,
    operationalization_schema,
    render_admission_context,
)
from .directive import (
    MAX_REQUIREMENT_CHARS,
    MODEL_CALLS_CEILING,
    AdmissionDirective,
)
from .door import (
    AdmissionInputs,
    AdmissionRefusedError,
    require_selected_candidate_for_admission,
)
from .gates import check_operationalization
from .preflight import (
    AdmissionCallPlan,
    AdmissionPreflightError,
    check_admission_coherence,
)
from .records import (
    CLAIM_KINDS,
    MAX_ARM_CHARS,
    MAX_ENCODINGS_PER_PREDICTION,
    MECHANICAL_READING,
    AdmissionRecord,
    GroundedSupport,
    OperationalPrediction,
    Requirement,
    RequirementSource,
    SupportSource,
)
from .store import (
    AdmissionConflictError,
    AdmissionIntegrityError,
    AdmissionStore,
)

__all__ = [
    "CLAIM_KINDS",
    "MAX_ARM_CHARS",
    "MAX_ENCODINGS_PER_PREDICTION",
    "MAX_REQUIREMENT_CHARS",
    "MECHANICAL_READING",
    "MODEL_CALLS_CEILING",
    "OPERATIONALIZATION_INSTRUCTION",
    "AdmissionBudgetError",
    "AdmissionCallPlan",
    "AdmissionConflictError",
    "AdmissionDirective",
    "AdmissionInputs",
    "AdmissionIntegrityError",
    "AdmissionPreflightError",
    "AdmissionRecord",
    "AdmissionRefusedError",
    "AdmissionRejectedError",
    "AdmissionRunResult",
    "AdmissionStore",
    "CandidateAdmitter",
    "GroundedSupport",
    "MappingRejection",
    "OperationalPrediction",
    "Requirement",
    "RequirementSource",
    "SupportSource",
    "build_field_texts",
    "check_admission_coherence",
    "check_operationalization",
    "encoded_metric",
    "operationalization_schema",
    "render_admission_context",
    "require_selected_candidate_for_admission",
]
