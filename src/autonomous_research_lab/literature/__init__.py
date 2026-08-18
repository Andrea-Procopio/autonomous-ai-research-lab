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

from .openalex import OpenAlexProvider
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
    RetrievedSearch,
    ScriptedLiteratureProvider,
    normalize_arxiv_id,
    normalize_doi,
)

__all__ = [
    "PAGE_SIZE_CEILING",
    "RESULT_CEILING",
    "AccessLevel",
    "LiteratureAuthenticationError",
    "LiteratureConfigurationError",
    "LiteratureProvider",
    "LiteratureProviderError",
    "LiteratureQuery",
    "LiteratureRateLimitError",
    "LiteratureSource",
    "LiteratureTimeoutError",
    "LiteratureTransportError",
    "MalformedLiteratureResponseError",
    "OpenAlexProvider",
    "RetrievedSearch",
    "ScriptedLiteratureProvider",
    "normalize_arxiv_id",
    "normalize_doi",
]
