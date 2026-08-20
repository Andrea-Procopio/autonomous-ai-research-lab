"""A funded research run: identity, authorization, and the ledger.

The analysis chain ends at admission, which produces one bare initial
:class:`~..core.state.ResearchState` holding propositions and no budget.
This package is what turns that seed into something a runtime may spend
against::

    admission record -> RunDirective + FundingAuthorization -> ResearchRun

The defect it exists to fix is an identity one. A state's content id
deliberately excludes its budget, so funding a persisted state by
replacing its budget produces different bytes under the same id, and the
append-only snapshot store refuses the write — correctly. Funding is
therefore *succession*: the funded state is a child of the admitted one,
with its own identity, and the admitted snapshot keeps its zero budget
forever.

What is spent afterwards is a ledger fact, not a field rewrite. Entries
are sequence-numbered, hash-chained, write-once, and idempotent by
charge id, so a run's balance survives a restart, a repeated charge
cannot debit twice, and two concurrent debits cannot overspend.

Binding negatives: a grant is authorization, never scientific standing —
``ADMITTED`` did not mean true, and ``FUNDED`` does not either; no
result, evidence, assessment, or claim is expressible anywhere in this
vocabulary; the admission artifacts are read and never written; and
:meth:`~..core.state.ResearchState.fund` is the only state mutator this
package may call, because every evolution after the funded state belongs
to orchestration's commit layer.
"""

from .authorization import (
    MAX_AUTHORITY_CHARS,
    MAX_GPU_HOURS,
    MAX_MODEL_TOKENS,
    MAX_USD,
    MAX_WALL_CLOCK_SECONDS,
    FundingAuthorization,
    UnauthorizedGrantError,
)
from .directive import MAX_LABEL_CHARS, RunDirective
from .door import (
    RunInputs,
    RunRefusedError,
    require_admitted_state_for_run,
)
from .integrity import (
    IntegrityIssue,
    IntegrityIssueKind,
    IntegrityReport,
    verify_run,
)
from .ledger import (
    BudgetLedger,
    LedgerConflictError,
    LedgerContentionError,
    LedgerIntegrityError,
    LedgerMismatchError,
)
from .preflight import RunPlan, RunPreflightError, check_funding_coherence
from .records import (
    MAX_REASON_CHARS,
    BudgetEntry,
    EntryKind,
    ResearchRun,
)
from .starter import RunStartResult, start_run
from .store import (
    ProgramConflictError,
    ProgramIntegrityError,
    ProgramStore,
)

__all__ = [
    "MAX_AUTHORITY_CHARS",
    "MAX_GPU_HOURS",
    "MAX_LABEL_CHARS",
    "MAX_MODEL_TOKENS",
    "MAX_REASON_CHARS",
    "MAX_USD",
    "MAX_WALL_CLOCK_SECONDS",
    "BudgetEntry",
    "BudgetLedger",
    "EntryKind",
    "FundingAuthorization",
    "IntegrityIssue",
    "IntegrityIssueKind",
    "IntegrityReport",
    "LedgerConflictError",
    "LedgerContentionError",
    "LedgerIntegrityError",
    "LedgerMismatchError",
    "ProgramConflictError",
    "ProgramIntegrityError",
    "ProgramStore",
    "ResearchRun",
    "RunDirective",
    "RunInputs",
    "RunPlan",
    "RunPreflightError",
    "RunRefusedError",
    "RunStartResult",
    "UnauthorizedGrantError",
    "check_funding_coherence",
    "require_admitted_state_for_run",
    "start_run",
    "verify_run",
]
