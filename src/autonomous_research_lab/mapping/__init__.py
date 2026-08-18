"""Evidence-grounded field mapping over the Task 5A literature corpus.

The chain this package implements, and nothing more::

    research brief / broad topic
      -> focused literature queries        (model-proposed, code-executed)
      -> Task 5A retrieval and replay
      -> relevance screening               (every verdict preserved)
      -> structured source-grounded extraction
      -> FieldMap
      -> ProblemInventory

Literature analysis is not experimentation: nothing here creates
``Evidence``, ``ResearchQuestion``, ``Hypothesis``, or any other
scientific-state proposition, and nothing here can reach
``ResearchState`` — both directions pinned by the structural tests. The
package depends on ``core``, ``literature``, and the provider seam
alone; candidate research-question and idea generation are Task 5C's,
reading these durable records.
"""

from .adequacy import (
    AdequacyMetrics,
    AdequacyReason,
    AdequacyReasonCode,
    AdequacyStatus,
    AdequacyThresholds,
    InadequateFieldMapError,
    MapAdequacyAssessment,
    ProblemSupport,
    SupportTier,
    assess_adequacy,
    require_adequate_for_idea_generation,
    support_tier,
)
from .brief import (
    REQUIRED_FAMILIES,
    QueryFamily,
    ResearchBrief,
    SourceEra,
    classify_era,
)
from .gates import (
    BANNED_COVERAGE_PHRASES,
    MappingRejection,
    accessible_text_of,
    check_extraction,
    check_field_map,
    check_inventory,
    check_queries,
    check_refined_queries,
    check_screening,
)
from .mapper import (
    EXTRACTION_SCHEMA,
    FIELD_MAP_SCHEMA,
    INVENTORY_SCHEMA,
    QUERY_SCHEMA,
    RETRIEVAL_STRATEGIES,
    SCREENING_SCHEMA,
    FieldMapper,
    MappingBudgetError,
    MappingContractError,
    MappingRejectedError,
    MappingRunResult,
)
from .records import (
    CLAIM_KINDS,
    CallProvenance,
    CoverageReport,
    DatasetAvailability,
    DatasetUse,
    ExtractionRecord,
    FieldMapRecord,
    GroupEntry,
    Limitation,
    LimitationKind,
    MappingRunRecord,
    ProblemEntry,
    ProblemInventoryRecord,
    ProblemKind,
    QueryExecution,
    RelationshipKind,
    ScreeningDecision,
    ScreeningRecord,
    SupportLocation,
    ThemeEntry,
    ThemeEra,
    ThemeRelationship,
)
from .store import MappingConflictError, MappingIntegrityError, MappingStore

__all__ = [
    "BANNED_COVERAGE_PHRASES",
    "CLAIM_KINDS",
    "EXTRACTION_SCHEMA",
    "FIELD_MAP_SCHEMA",
    "INVENTORY_SCHEMA",
    "QUERY_SCHEMA",
    "REQUIRED_FAMILIES",
    "RETRIEVAL_STRATEGIES",
    "SCREENING_SCHEMA",
    "AdequacyMetrics",
    "AdequacyReason",
    "AdequacyReasonCode",
    "AdequacyStatus",
    "AdequacyThresholds",
    "CallProvenance",
    "CoverageReport",
    "DatasetAvailability",
    "DatasetUse",
    "ExtractionRecord",
    "FieldMapRecord",
    "FieldMapper",
    "GroupEntry",
    "InadequateFieldMapError",
    "Limitation",
    "LimitationKind",
    "MapAdequacyAssessment",
    "MappingBudgetError",
    "MappingConflictError",
    "MappingContractError",
    "MappingIntegrityError",
    "MappingRejectedError",
    "MappingRejection",
    "MappingRunRecord",
    "MappingRunResult",
    "MappingStore",
    "ProblemEntry",
    "ProblemInventoryRecord",
    "ProblemKind",
    "ProblemSupport",
    "QueryExecution",
    "QueryFamily",
    "RelationshipKind",
    "ResearchBrief",
    "ScreeningDecision",
    "ScreeningRecord",
    "SourceEra",
    "SupportLocation",
    "SupportTier",
    "ThemeEntry",
    "ThemeEra",
    "ThemeRelationship",
    "accessible_text_of",
    "assess_adequacy",
    "check_extraction",
    "check_field_map",
    "check_inventory",
    "check_queries",
    "check_refined_queries",
    "check_screening",
    "classify_era",
    "require_adequate_for_idea_generation",
    "support_tier",
]
