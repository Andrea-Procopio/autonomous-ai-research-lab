"""The literature-retrieval boundary: one narrow seam between the lab and a
scholarly index.

Later stages need to know what external papers report; nothing else in the
system should know that a scholarly API exists, let alone which one. This
module is the whole contract:

    LiteratureQuery  ->  LiteratureProvider.search()  ->  RetrievedSearch

An adapter translates a bounded query into whatever its API wants, pages
through the reply within the query's budgets, and translates each raw work
into a :class:`LiteratureSource`. **No provider object crosses this
boundary.** Every field is a string, a number, a tuple, or a mapping of
those, so a caller cannot depend on a vendor payload shape, and a recorded
source stays readable after the API is gone.

Literature is not evidence
--------------------------

A :class:`LiteratureSource` describes what an external paper *reports*. It
is not an ``ExperimentResult``, not ``Evidence``, and not proof that any
claim is true; nothing in this package can reach ``ResearchState``, and the
structural tests pin that. The ``access_level`` field records how much of
the paper was actually retrieved — metadata alone, or metadata plus
abstract — so no later stage can claim to have read text that was never
fetched. Task 5A retrieves at most abstracts; ``FULL_TEXT`` exists in the
vocabulary so the level can be expressed honestly if a later stage earns it.

Bounded by construction
-----------------------

Every query carries explicit result and page-size budgets, validated at
construction against hard ceilings, so a broad topic cannot grow into an
unbounded crawl. Failures mirror the model-provider seam's taxonomy: local
misconfiguration, rejected credential or access, rate limiting, timeout,
transport, and malformed reply are distinct types a caller can act on. A
failed retrieval is an infrastructure event, never a scientific outcome.
"""

from __future__ import annotations

import datetime as _dt
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from ..core.ids import content_id
from ..core.types import freeze_mapping

RESULT_CEILING: Final = 1_000
"""Hard upper bound on ``max_results`` for a single query. Field mapping
needs recent, relevant work, not a crawl; anything larger belongs to a bulk
snapshot, not this seam."""

PAGE_SIZE_CEILING: Final = 100
"""Hard upper bound on one page. Matches the live provider's documented
maximum (OpenAlex ``per_page`` is 1-100) and keeps any single reply small."""

DEFAULT_TIMEOUT_SECONDS: Final = 30.0
"""Per-page-request deadline. Every network exchange has a finite timeout,
for the same reason every model call and every experiment job does."""


# -- failures -----------------------------------------------------------------


class LiteratureProviderError(RuntimeError):
    """Base class for every literature-retrieval failure.

    A failed retrieval is an infrastructure event, never a scientific
    result and never a statement about the literature itself.
    """


class LiteratureConfigurationError(LiteratureProviderError):
    """The caller's own configuration is missing or malformed — an
    unusable base URL, an unsupported scheme — detected before any request
    is made. Permanent until a human fixes it; no retry can help."""


class LiteratureAuthenticationError(LiteratureProviderError):
    """The provider refused the caller's credential or access (the
    documented HTTP 403 refusal, or a 401). The verdict came from the
    remote side; retrying with the same credential fails identically."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_error = provider_error
        """The provider's own error identifier, when it supplied one."""


class LiteratureRateLimitError(LiteratureProviderError):
    """The provider is throttling — a per-second limit or an exhausted
    daily budget. Carries the wait it asked for, when it gave one, so a
    caller can back off on fact rather than on guesswork."""

    def __init__(
        self, message: str, *, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LiteratureTimeoutError(LiteratureProviderError):
    """A page request exceeded its deadline. Distinguished from transport
    failure because retrying a timeout costs a full request again."""

    def __init__(self, message: str, *, timeout_seconds: float) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class LiteratureTransportError(LiteratureProviderError):
    """The provider could not be reached, or refused the request: connection
    failures, server errors, bad requests, and missing resources — cases
    where no usable reply was produced and the fault is not specifically a
    timeout, a rate limit, or a rejected credential."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_error = provider_error


class MalformedLiteratureResponseError(LiteratureProviderError):
    """The provider replied, but the reply is unusable: not valid UTF-8,
    not valid JSON, or not the shape the documented protocol promises."""


# -- identifier canonicalization ----------------------------------------------


_DOI_PREFIXES: Final = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
)

#: New-style (2007+) arXiv id: YYMM.NNNNN. Old-style: archive/YYMMNNN with an
#: optional subject class, e.g. ``math.gt/0309136``.
_ARXIV_NEW: Final = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_OLD: Final = re.compile(r"^[a-z-]+(\.[a-z]{2})?/\d{7}$")
_ARXIV_VERSION: Final = re.compile(r"v\d+$")


def normalize_doi(value: str | None) -> str | None:
    """The canonical form of a DOI, or ``None`` when the input is not one.

    Resolver prefixes (``https://doi.org/`` and kin) and a ``doi:`` label
    are stripped; the remainder is lower-cased, because DOIs are
    case-insensitive by specification. Anything that does not then look
    like a DOI (``10.<registrant>/<suffix>``) yields ``None`` — a mangled
    identifier must not silently become a plausible-looking key.
    """
    if value is None:
        return None
    candidate = value.strip()
    lowered = candidate.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            lowered = lowered[len(prefix) :]
            break
    if not lowered.startswith("10.") or "/" not in lowered:
        return None
    return lowered


def normalize_arxiv_id(value: str | None) -> str | None:
    """The canonical form of an arXiv id, or ``None`` when the input is not
    one.

    Accepts the bare id, an ``arXiv:`` label, an ``arxiv.org/abs/`` or
    ``/pdf/`` URL, and the arXiv DOI form (``10.48550/arxiv.<id>``). The
    id is lower-cased and any ``vN`` version suffix is dropped: versions
    are revisions of one work, and work identity is what deduplication
    needs. Anything that does not then match the old- or new-style grammar
    yields ``None`` rather than becoming a fabricated key.
    """
    if value is None:
        return None
    candidate = value.strip().lower()
    if "arxiv.org/" in candidate:
        _, _, tail = candidate.partition("arxiv.org/")
        for route in ("abs/", "pdf/"):
            if tail.startswith(route):
                tail = tail[len(route) :]
                break
        else:
            return None
        candidate = tail.removesuffix(".pdf")
    candidate = candidate.removeprefix("arxiv:")
    candidate = candidate.removeprefix("10.48550/arxiv.")
    candidate = _ARXIV_VERSION.sub("", candidate).strip("/")
    if _ARXIV_NEW.match(candidate) or _ARXIV_OLD.match(candidate):
        return candidate
    return None


# -- the query ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiteratureQuery:
    """One bounded literature search, described in provider-neutral terms.

    Dates are inclusive ISO ``YYYY-MM-DD`` bounds on publication date; an
    empty string leaves that side unbounded. ``max_results`` and
    ``per_page`` are validated against hard ceilings at construction, so a
    query that could grow without bound cannot be built at all.
    """

    text: str
    from_date: str = ""
    to_date: str = ""
    per_page: int = 25
    max_results: int = 100
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Per-page-request deadline, not a whole-search bound: the search's
    total time is bounded by this times the (budget-limited) page count."""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text must be non-empty")
        for label, value in (("from_date", self.from_date), ("to_date", self.to_date)):
            if value:
                try:
                    _dt.date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{label} must be an ISO date (YYYY-MM-DD): {value!r}"
                    ) from exc
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError(
                f"from_date {self.from_date} is after to_date {self.to_date}"
            )
        if not 1 <= self.per_page <= PAGE_SIZE_CEILING:
            raise ValueError(
                f"per_page must be in 1..{PAGE_SIZE_CEILING}, got {self.per_page}"
            )
        if not 1 <= self.max_results <= RESULT_CEILING:
            raise ValueError(
                f"max_results must be in 1..{RESULT_CEILING}, got {self.max_results}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def fingerprint(self) -> str:
        """A content id over everything that determines the result set.

        ``timeout_seconds`` deliberately does not participate: unlike a
        model call, where the deadline shapes what can be generated, a
        search's result set is decided server-side — two queries differing
        only in patience ask for the same results, and should replay from
        the same cached search.
        """
        return content_id(
            "litq",
            self.text,
            self.from_date,
            self.to_date,
            self.per_page,
            self.max_results,
        )


# -- the normalized source ----------------------------------------------------


class AccessLevel(StrEnum):
    """How much of the paper was actually retrieved. Preserved on every
    record so later stages cannot claim to have read more than was
    fetched."""

    METADATA = "metadata"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"


@dataclass(frozen=True, slots=True)
class LiteratureSource:
    """One paper, as one provider reported it at one retrieval.

    A snapshot, not a living record: the id is content-derived over every
    field, so the same reported metadata is the same source wherever it
    was retrieved, and a re-retrieval that reports different metadata (a
    grown citation count, a corrected title) is a *different* snapshot.
    Work-level identity — "are these snapshots the same paper?" — is the
    deduplication layer's question, answered by the canonical external
    identifiers, never by silently rewriting a record.

    Optional metadata the provider did not report stays ``None`` or empty:
    an absent abstract is absent, not an empty string pretending to be one.
    """

    provider: str
    provider_id: str
    title: str | None
    authors: tuple[str, ...]
    publication_date: str | None
    publication_year: int | None
    venue: str | None
    work_type: str | None
    abstract: str | None
    doi: str | None
    arxiv_id: str | None
    provider_url: str
    landing_page_url: str | None
    pdf_url: str | None
    cited_by_count: int | None
    referenced_work_ids: tuple[str, ...]
    access_level: AccessLevel
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("source must name its provider")
        if not self.provider_id.strip():
            raise ValueError("source must carry the provider's identifier")
        if self.abstract is not None and not self.abstract.strip():
            raise ValueError("an absent abstract must be None, not blank text")
        if self.abstract is None and self.access_level is not AccessLevel.METADATA:
            raise ValueError(
                f"access level {self.access_level.value!r} claims text that "
                f"was not retrieved; without an abstract the level is "
                f"'metadata'"
            )
        if self.abstract is not None and self.access_level is AccessLevel.METADATA:
            raise ValueError(
                "a source with an abstract must say so: access level "
                "'metadata' understates what was retrieved"
            )
        if not self.id:
            object.__setattr__(
                self,
                "id",
                content_id(
                    "lit",
                    self.provider,
                    self.provider_id,
                    self.title,
                    self.authors,
                    self.publication_date,
                    self.publication_year,
                    self.venue,
                    self.work_type,
                    self.abstract,
                    self.doi,
                    self.arxiv_id,
                    self.provider_url,
                    self.landing_page_url,
                    self.pdf_url,
                    self.cited_by_count,
                    self.referenced_work_ids,
                    self.access_level,
                ),
            )


# -- the completed retrieval --------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievedSearch:
    """One completed bounded search, in terms nothing downstream has to
    decode: the provider parameters actually sent (credential-free by
    contract), what the provider said about the result set, per-page
    response identifiers when the provider supplied any, rate-limit
    observations, and the normalized sources in provider order.
    """

    provider: str
    retrieved_at: str
    """UTC ISO-8601 timestamp taken when the retrieval began."""

    request_params: Mapping[str, str]
    total_count: int | None
    """The provider's own count of everything matching, when reported —
    usually far larger than the bounded slice actually retrieved."""

    pages_fetched: int
    page_identifiers: tuple[str, ...]
    """One entry per page: the response identifier when the provider
    supplied one, empty string otherwise — absence stays visible."""

    rate_limit: Mapping[str, str]
    """Provider-prefixed rate/budget observations (credits used, remaining
    quota), verbatim strings. Never contains a credential."""

    truncated: bool
    """Whether more matching results existed beyond the query's budget."""

    sources: tuple[LiteratureSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_params", freeze_mapping(self.request_params)
        )
        object.__setattr__(self, "rate_limit", freeze_mapping(self.rate_limit))
        if not self.provider.strip():
            raise ValueError("a retrieval must name its provider")
        if self.pages_fetched < 1:
            raise ValueError("a completed retrieval fetched at least one page")
        if len(self.page_identifiers) != self.pages_fetched:
            raise ValueError(
                "page_identifiers must carry one entry per fetched page"
            )


# -- the provider interface ---------------------------------------------------


class LiteratureProvider(ABC):
    """The one interface the lab uses to reach a scholarly index.

    Implementations translate to and from a concrete API and raise the
    :class:`LiteratureProviderError` family on failure. They never touch
    ``ResearchState``: a retrieved paper is a description of external work
    until some later, separate stage decides what to make of it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable provider identifier, recorded on every source and search."""

    @abstractmethod
    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        """Perform ``query`` within its budgets and return the retrieval.

        Raises :class:`LiteratureConfigurationError`,
        :class:`LiteratureAuthenticationError`,
        :class:`LiteratureRateLimitError`, :class:`LiteratureTimeoutError`,
        :class:`LiteratureTransportError`, or
        :class:`MalformedLiteratureResponseError`.
        """


# -- the deterministic test provider ------------------------------------------


class ScriptedLiteratureProvider(LiteratureProvider):
    """A provider with no network and no clock, mirroring the model seam's
    ``FakeModelProvider``: retrievals are served from a script, in order,
    and every received query is recorded for assertion."""

    def __init__(
        self,
        outcomes: Sequence[RetrievedSearch | LiteratureProviderError],
        *,
        name: str = "scripted",
    ) -> None:
        self._outcomes = tuple(outcomes)
        self._name = name
        self._queries: list[LiteratureQuery] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def queries(self) -> tuple[LiteratureQuery, ...]:
        return tuple(self._queries)

    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        index = len(self._queries)
        self._queries.append(query)
        if index >= len(self._outcomes):
            raise MalformedLiteratureResponseError(
                f"scripted provider was scripted for {len(self._outcomes)} "
                f"search(es), received search {index + 1}"
            )
        outcome = self._outcomes[index]
        if isinstance(outcome, LiteratureProviderError):
            raise outcome
        return outcome
