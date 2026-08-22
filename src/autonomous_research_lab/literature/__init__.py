"""Bounded, reproducible literature retrieval.

The chain this package implements, and nothing more::

    bounded literature query
      -> real scholarly API (one adapter, OpenAlex)
      -> normalized source records
      -> deterministic deduplication
      -> durable search provenance
      -> reproducible local corpus

Literature records describe what external papers *report*. They are not
``ExperimentResult``, not ``Evidence``, and not proof that any claim is
true; nothing here touches ``ResearchState``, generates hypotheses, or
judges novelty — those are later, separate stages that will read this
package's durable records. Everything here depends on ``core`` only, and
nothing else in the package depends on this; both directions are pinned by
the structural tests.
"""

from .arxiv import fetch_abstract, summary_from
from .corpus import CorpusSearchResult, LiteratureCorpus
from .dedup import (
    DeduplicationReport,
    DuplicateConflict,
    DuplicateGroup,
    deduplicate,
)
from .openalex import OpenAlexProvider
from .resolution import AccessResolvingProvider
from .retrieval import (
    PAGE_SIZE_CEILING,
    RESULT_CEILING,
    AccessLevel,
    LiteratureAuthenticationError,
    LiteratureConfigurationError,
    LiteratureProvider,
    LiteratureProviderError,
    LiteratureQuery,
    LiteratureRateLimitError,
    LiteratureSource,
    LiteratureTimeoutError,
    LiteratureTransportError,
    MalformedLiteratureResponseError,
    ResultOrdering,
    RetrievedSearch,
    ScriptedLiteratureProvider,
    normalize_arxiv_id,
    normalize_doi,
)
from .store import (
    LiteratureConflictError,
    LiteratureCorpusLimitError,
    LiteratureIntegrityError,
    LiteratureSearchRecord,
    LiteratureStore,
    search_record_from,
)

__all__ = [
    "PAGE_SIZE_CEILING",
    "RESULT_CEILING",
    "AccessLevel",
    "AccessResolvingProvider",
    "CorpusSearchResult",
    "DeduplicationReport",
    "DuplicateConflict",
    "DuplicateGroup",
    "LiteratureAuthenticationError",
    "LiteratureConfigurationError",
    "LiteratureConflictError",
    "LiteratureCorpus",
    "LiteratureCorpusLimitError",
    "LiteratureIntegrityError",
    "LiteratureProvider",
    "LiteratureProviderError",
    "LiteratureQuery",
    "LiteratureRateLimitError",
    "LiteratureSearchRecord",
    "LiteratureSource",
    "LiteratureStore",
    "LiteratureTimeoutError",
    "LiteratureTransportError",
    "MalformedLiteratureResponseError",
    "OpenAlexProvider",
    "ResultOrdering",
    "RetrievedSearch",
    "ScriptedLiteratureProvider",
    "deduplicate",
    "fetch_abstract",
    "normalize_arxiv_id",
    "normalize_doi",
    "search_record_from",
    "summary_from",
]
