"""OpenAlex adapter: the one concrete literature provider behind the seam.

Standard-library HTTP only — the package keeps its zero-dependency promise,
and every vendor-shaped object (URLs, wire payloads, ``http.client`` types,
raw work dicts) stays inside this module.

Why OpenAlex
------------

Chosen over Semantic Scholar and kin because it is the smallest defensible
source of recent-paper coverage with stable identifiers: fully open (CC0)
metadata, no credential required for basic use, provider-normalized DOIs,
durable ``W…`` work ids, cursor paging, and an explicit sort that makes
bounded retrieval reproducible. Semantic Scholar's dependable rate tier
requires an API-key application, which a zero-credential foundation should
not assume.

Wire contract
-------------

Established from the official documentation (help.openalex.org/api, the
post-2026 revision) and from live responses captured on 2026-08-18.
Observed directly: the success envelope (``meta`` with ``count`` and
``next_cursor``, ``results`` as a list of work objects), cursor paging with
``sort=publication_date:desc``, the credit-based rate-limit headers
(``x-ratelimit-credits-used``, ``x-ratelimit-remaining``, …), the arXiv
identifier shapes (a ``10.48550/arxiv.<id>`` DOI, or an
``arxiv.org/abs/…`` landing/pdf URL), and a 404 whose body is **HTML, not
the documented JSON envelope** — so error-body parsing must degrade to the
status code. Taken from the documentation without live capture: the
``{"error", "message"}`` JSON error envelope, the 403 access refusal, 429
for both the per-second limit and the exhausted daily credit budget, and
the retry guidance (exponential backoff for 429/5xx; ``Retry-After`` is in
the exposed-header list).

* ``GET {base_url}/works?search=…&filter=from_publication_date:…,
  to_publication_date:…&per_page=…&sort=publication_date:desc&cursor=…``
* Paging: start at ``cursor=*``; continue with ``meta.next_cursor``; done
  when the cursor is null or ``results`` is empty. ``per_page`` is 1-100.
* Ordering: ``publication_date:desc`` is requested explicitly because the
  default relevance ranking is not stable across index updates. The
  provider's tiebreak among same-date works is unspecified, which is one
  reason the durable search record preserves the returned order verbatim.
* Abstracts arrive as ``abstract_inverted_index`` (word -> positions) and
  are reconstructed locally; a work without one yields access level
  ``metadata``.

Retry policy, from the documented semantics only: at most **one** retry
per page request, only for 429 and 5xx (the two families the documentation
says to back off on), waiting the server's ``Retry-After`` when it gives a
usable one and the documented first backoff step (2 s) otherwise. A server
wait longer than :data:`MAX_RETRY_WAIT_SECONDS` is surfaced as the
rate-limit error instead of being slept through inside a call.

The optional API key is read from the environment (``OPENALEX_API_KEY``)
at request time, sent only as an ``Authorization: Bearer`` header — never
as a query parameter, so it can never appear in a recorded request — and
never logged or stored. No key is required: anonymous use has a documented
daily credit budget (observed: 1000 credits/day, 10 credits per search
page).

The whole page exchange after DNS resolution runs under one wall-clock
deadline enforced by a watchdog that closes the connection at expiry,
mirroring ``runtime/muse.py`` (see that module for the live incident that
motivated it); the per-socket-operation timeout remains as a first fence,
and ``getaddrinfo`` remains the named uninterruptible residual.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import math
import os
import socket
import ssl
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final, Protocol

from .retrieval import (
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
    normalize_arxiv_id,
    normalize_doi,
)

DEFAULT_BASE_URL: Final = "https://api.openalex.org"
KEY_ENV_VAR: Final = "OPENALEX_API_KEY"
PROVIDER_NAME: Final = "openalex"

SORT: Final = "publication_date:desc"
"""The explicit deterministic-where-supported ordering; relevance ranking
is not stable across index updates and is never requested."""

MAX_RETRY_WAIT_SECONDS: Final = 30.0
_BACKOFF_SECONDS: Final = 2.0
"""The documented first exponential-backoff step (2^1) — the only one this
adapter takes, since it retries at most once."""

_USER_AGENT: Final = "autonomous-research-lab/0.0.1 (literature retrieval)"
_CHUNK_BYTES: Final = 64 * 1024
_MAX_BODY_BYTES: Final = 8 * 1024 * 1024
"""A 100-work page with abstracts is under 2 MiB; anything past this is a
fault to classify, not a result to buffer."""

_MAX_ERROR_BODY_BYTES: Final = 256 * 1024

#: Rate-limit headers copied into the retrieval's ``rate_limit`` mapping,
#: ``openalex:``-prefixed. Observed live on 2026-08-18.
_RATE_LIMIT_HEADERS: Final = (
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-cost-usd",
    "x-ratelimit-remaining-usd",
)


class OpenAlexProvider(LiteratureProvider):
    """OpenAlex ``/works`` search adapter, synchronous, cursor-paged."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._sleep = sleep

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def search(self, query: LiteratureQuery) -> RetrievedSearch:
        retrieved_at = _utc_now()
        base_params = _base_params(query)
        sources: list[LiteratureSource] = []
        page_identifiers: list[str] = []
        credits_used = 0
        rate_limit: dict[str, str] = {}
        total_count: int | None = None
        truncated = False
        cursor = "*"

        while True:
            remaining = query.max_results - len(sources)
            page_params = dict(base_params)
            page_params["per_page"] = str(min(query.per_page, remaining))
            page_params["cursor"] = cursor
            body, headers = self._fetch_page(page_params, query.timeout_seconds)

            meta = body.get("meta")
            if not isinstance(meta, Mapping):
                raise MalformedLiteratureResponseError(
                    "the reply carries no meta object"
                )
            results = body.get("results")
            if not isinstance(results, list):
                raise MalformedLiteratureResponseError(
                    "the reply carries no results list"
                )
            if total_count is None:
                total_count = _int_or_none(meta.get("count"))

            page_identifiers.append(headers.get("cf-ray", ""))
            credits_used += _int_from_header(headers, "x-ratelimit-credits-used")
            for header in _RATE_LIMIT_HEADERS:
                if header in headers:
                    rate_limit[f"openalex:{header}"] = headers[header]

            for raw in results[:remaining]:
                sources.append(_normalize_work(raw))
            next_cursor = meta.get("next_cursor")
            has_more = (
                isinstance(next_cursor, str) and bool(next_cursor) and bool(results)
            )
            if len(sources) >= query.max_results:
                truncated = has_more or len(results) > remaining
                break
            if not isinstance(next_cursor, str) or not has_more:
                break
            cursor = next_cursor

        rate_limit["openalex:credits_used_total"] = str(credits_used)
        return RetrievedSearch(
            provider=PROVIDER_NAME,
            retrieved_at=retrieved_at,
            request_params=base_params | {"cursor": "*"},
            total_count=total_count,
            pages_fetched=len(page_identifiers),
            page_identifiers=tuple(page_identifiers),
            rate_limit=rate_limit,
            truncated=truncated,
            sources=tuple(sources),
        )

    # -- one page, with the documented bounded retry --------------------------

    def _fetch_page(
        self, params: Mapping[str, str], timeout: float
    ) -> tuple[Mapping[str, object], Mapping[str, str]]:
        try:
            return self._exchange(params, timeout)
        except (LiteratureRateLimitError, LiteratureTransportError) as exc:
            wait = _retry_wait(exc)
            if wait is None:
                raise
            self._sleep(wait)
            return self._exchange(params, timeout)

    def _exchange(
        self, params: Mapping[str, str], timeout: float
    ) -> tuple[Mapping[str, object], Mapping[str, str]]:
        """One GET under one wall-clock deadline, entered after DNS
        resolution. ``http.client`` types enter and never leave."""
        target = f"/works?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
        api_key = _api_key()
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        deadline = time.monotonic() + timeout
        connection = _connect(self._base_url, timeout)
        watchdog = _DeadlineWatchdog(connection, deadline)
        try:
            with watchdog:
                connection.request("GET", target, headers=headers)
                reply = connection.getresponse()
                reply_headers = _lowered(reply.headers.items())
                if not 200 <= reply.status < 300:
                    raise _error_for_status(
                        reply.status,
                        reply_headers,
                        _drain_error_body(reply, deadline=deadline),
                    )
                raw = _read_bounded(reply, deadline=deadline, timeout=timeout)
        except LiteratureProviderError:
            raise
        except TimeoutError as exc:
            raise LiteratureTimeoutError(
                f"no reply from the OpenAlex endpoint within {timeout}s",
                timeout_seconds=timeout,
            ) from exc
        except (
            http.client.HTTPException,
            OSError,
            ValueError,
            AttributeError,
        ) as exc:
            # A watchdog close surfaces as whatever the interrupted
            # primitive raises; when the deadline caused it, the deadline
            # is the diagnosis. (Mirrors runtime/muse.py.)
            if watchdog.fired:
                raise LiteratureTimeoutError(
                    f"no reply from the OpenAlex endpoint within {timeout}s",
                    timeout_seconds=timeout,
                ) from exc
            if isinstance(exc, http.client.HTTPException):
                raise LiteratureTransportError(
                    f"malformed HTTP exchange with the OpenAlex endpoint: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if isinstance(exc, OSError):
                raise LiteratureTransportError(
                    f"could not reach the OpenAlex endpoint: {exc}"
                ) from exc
            raise
        finally:
            connection.close()

        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MalformedLiteratureResponseError(
                f"the reply is not valid JSON: {exc}"
            ) from exc
        if not isinstance(body, Mapping):
            raise MalformedLiteratureResponseError(
                f"the reply is not a JSON object: {type(body).__name__}"
            )
        return body, reply_headers


def _base_params(query: LiteratureQuery) -> dict[str, str]:
    """The credential-free provider parameters that define the search —
    exactly what a durable search record may store."""
    params = {"search": query.text}
    filters = []
    if query.from_date:
        filters.append(f"from_publication_date:{query.from_date}")
    if query.to_date:
        filters.append(f"to_publication_date:{query.to_date}")
    if filters:
        params["filter"] = ",".join(filters)
    params["sort"] = SORT
    params["per_page"] = str(query.per_page)
    return params


def _retry_wait(exc: LiteratureProviderError) -> float | None:
    """The single documented-semantics retry: how long to wait, or ``None``
    when the failure must not be retried at all."""
    if isinstance(exc, LiteratureRateLimitError):
        wait = exc.retry_after_seconds
        if wait is not None and wait > MAX_RETRY_WAIT_SECONDS:
            return None
        return wait if wait is not None else _BACKOFF_SECONDS
    if isinstance(exc, LiteratureTransportError):
        status = exc.status_code
        if status is not None and status >= 500:
            return _BACKOFF_SECONDS
    return None


def _api_key() -> str | None:
    """The optional credential, read at request time and held only on the
    stack. ``None`` means anonymous use, which OpenAlex supports."""
    value = os.environ.get(KEY_ENV_VAR, "").strip()
    return value or None


def _utc_now() -> str:
    """Module-level on purpose: the deterministic test seam for the
    retrieval timestamp."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


# -- work normalization -------------------------------------------------------


def _normalize_work(raw: object) -> LiteratureSource:
    """One raw OpenAlex work as a neutral :class:`LiteratureSource`.

    Strict about identity (a work without a usable ``W…`` id is a
    malformed reply), permissive about everything optional: metadata the
    provider did not report stays ``None``/empty rather than becoming a
    fabricated default.
    """
    if not isinstance(raw, Mapping):
        raise MalformedLiteratureResponseError(
            f"a result entry is not an object: {type(raw).__name__}"
        )
    provider_url = _str_or_none(raw.get("id"))
    provider_id = _openalex_tail(provider_url)
    if provider_url is None or provider_id is None:
        raise MalformedLiteratureResponseError(
            "a result entry carries no usable OpenAlex work id"
        )

    primary = raw.get("primary_location")
    primary = primary if isinstance(primary, Mapping) else {}
    source = primary.get("source")
    source = source if isinstance(source, Mapping) else {}

    doi = normalize_doi(_str_or_none(raw.get("doi")))
    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
    return LiteratureSource(
        provider=PROVIDER_NAME,
        provider_id=provider_id,
        title=_str_or_none(raw.get("title")) or _str_or_none(raw.get("display_name")),
        authors=_authors(raw.get("authorships")),
        publication_date=_str_or_none(raw.get("publication_date")),
        publication_year=_int_or_none(raw.get("publication_year")),
        venue=_str_or_none(source.get("display_name")),
        work_type=_str_or_none(raw.get("type")),
        abstract=abstract,
        doi=doi,
        arxiv_id=_arxiv_id(doi, raw, primary),
        provider_url=provider_url,
        landing_page_url=_str_or_none(primary.get("landing_page_url")),
        pdf_url=_str_or_none(primary.get("pdf_url")),
        cited_by_count=_int_or_none(raw.get("cited_by_count")),
        referenced_work_ids=_referenced_ids(raw.get("referenced_works")),
        access_level=(
            AccessLevel.ABSTRACT if abstract is not None else AccessLevel.METADATA
        ),
    )


def _openalex_tail(url: str | None) -> str | None:
    """The bare ``W…`` id from an ``https://openalex.org/W…`` URL."""
    if url is None:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.startswith("W") and len(tail) > 1 else None


def _str_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _int_from_header(headers: Mapping[str, str], name: str) -> int:
    try:
        return int(headers.get(name, ""))
    except ValueError:
        return 0


def _authors(value: object) -> tuple[str, ...]:
    """Author display names in listed order. ``display_name`` when the
    author is disambiguated, the raw byline string otherwise; entries with
    neither are skipped rather than invented."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    names = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        author = entry.get("author")
        author = author if isinstance(author, Mapping) else {}
        name = _str_or_none(author.get("display_name")) or _str_or_none(
            entry.get("raw_author_name")
        )
        if name is not None:
            names.append(name)
    return tuple(names)


def _referenced_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    tails = []
    for entry in value:
        tail = _openalex_tail(entry if isinstance(entry, str) else None)
        if tail is not None:
            tails.append(tail)
    return tuple(tails)


def _arxiv_id(
    doi: str | None, raw: Mapping[str, object], primary: Mapping[str, object]
) -> str | None:
    """The arXiv id, from either observed shape: the ``10.48550/arxiv.…``
    DOI, or an ``arxiv.org`` landing/pdf URL on any location."""
    from_doi = normalize_arxiv_id(doi)
    if from_doi is not None:
        return from_doi
    locations: list[Mapping[str, object]] = [primary]
    listed = raw.get("locations")
    if isinstance(listed, Sequence) and not isinstance(listed, (str, bytes)):
        locations.extend(loc for loc in listed if isinstance(loc, Mapping))
    for location in locations:
        for key in ("landing_page_url", "pdf_url"):
            candidate = normalize_arxiv_id(_str_or_none(location.get(key)))
            if candidate is not None:
                return candidate
    return None


def _reconstruct_abstract(index: object) -> str | None:
    """The abstract from OpenAlex's inverted index (word -> positions),
    words joined by single spaces in position order. Original whitespace
    is not recoverable — a known limitation, not a defect. A malformed
    index yields ``None``: no partial text masquerading as the abstract.
    """
    if not isinstance(index, Mapping) or not index:
        return None
    placed: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(word, str):
            return None
        if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
            return None
        for position in positions:
            if not isinstance(position, int) or isinstance(position, bool):
                return None
            placed.append((position, word))
    placed.sort(key=lambda pair: pair[0])
    text = " ".join(word for _, word in placed).strip()
    return text or None


# -- the connection and its deadline (mirrors runtime/muse.py) ----------------


def _connect(base_url: str, timeout: float) -> http.client.HTTPConnection:
    """One connection for one page request, scheme-dispatched. ``https``
    gets certificate verification; plain ``http`` is kept for loopback
    test servers. Module-level on purpose: the deterministic test seam."""
    split = urllib.parse.urlsplit(base_url)
    if split.scheme == "https":
        return http.client.HTTPSConnection(
            split.netloc, timeout=timeout, context=ssl.create_default_context()
        )
    if split.scheme == "http":
        return http.client.HTTPConnection(split.netloc, timeout=timeout)
    raise LiteratureConfigurationError(
        f"unsupported OpenAlex base URL scheme {split.scheme!r}: "
        f"expected http or https"
    )


class _DeadlineWatchdog:
    """Closes the connection at the absolute deadline, so a blocked read
    raises immediately instead of restarting a fresh per-operation
    timeout. Mirrors the muse adapter's watchdog; see runtime/muse.py for
    the live stall that motivated it."""

    def __init__(
        self, connection: http.client.HTTPConnection, deadline: float
    ) -> None:
        self._connection = connection
        self._fired = threading.Event()
        self._timer = threading.Timer(
            max(0.0, deadline - time.monotonic()), self._expire
        )
        self._timer.daemon = True

    def __enter__(self) -> _DeadlineWatchdog:
        self._timer.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._timer.cancel()

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def _expire(self) -> None:
        self._fired.set()
        sock = getattr(self._connection, "sock", None)
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._connection.close()


class _Reader(Protocol):
    def read(self, amount: int, /) -> bytes: ...


def _read_bounded(reply: _Reader, *, deadline: float, timeout: float) -> bytes:
    """The whole success body, bounded in both time and size; each
    iteration reads with ``read1`` so the deadline check runs between
    every socket operation. Mirrors runtime/muse.py."""
    read_one = getattr(reply, "read1", None)
    chunks = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LiteratureTimeoutError(
                f"the reply body did not complete within {timeout}s",
                timeout_seconds=timeout,
            )
        _tighten_timeout(reply, remaining)
        chunk: bytes = (
            read_one(_CHUNK_BYTES)
            if callable(read_one)
            else reply.read(_CHUNK_BYTES)
        )
        if not chunk:
            return bytes(chunks)
        chunks += chunk
        if len(chunks) > _MAX_BODY_BYTES:
            raise MalformedLiteratureResponseError(
                f"the reply body exceeds {_MAX_BODY_BYTES} bytes"
            )


def _tighten_timeout(reply: object, remaining: float) -> None:
    fp = getattr(reply, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    settimeout = getattr(sock, "settimeout", None)
    if callable(settimeout):
        with contextlib.suppress(OSError):
            settimeout(remaining)


def _drain_error_body(reply: _Reader, *, deadline: float) -> bytes:
    """The error body, best effort and bounded in size and time; a body
    that cannot be read degrades to the status-code fallback."""
    read_one = getattr(reply, "read1", None)
    chunks = bytearray()
    try:
        while len(chunks) <= _MAX_ERROR_BODY_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _tighten_timeout(reply, remaining)
            chunk: bytes = (
                read_one(_CHUNK_BYTES)
                if callable(read_one)
                else reply.read(_CHUNK_BYTES)
            )
            if not chunk:
                break
            chunks += chunk
    except (
        TimeoutError,
        OSError,
        ValueError,
        AttributeError,
        http.client.HTTPException,
    ):
        pass
    return bytes(chunks)


# -- error translation --------------------------------------------------------


def _error_for_status(
    status: int, headers: Mapping[str, str], raw: bytes
) -> LiteratureProviderError:
    """The documented statuses mapped onto the typed hierarchy: 401/403 is
    a refused credential or refused access, 429 is throttling (either the
    per-second limit or the exhausted daily credit budget) with the
    server's own ``Retry-After`` when usable, and everything else that
    produced no usable reply — 400, the observed HTML 404, 5xx — is a
    transport-level failure carrying the status and the provider's error
    label when the body was the documented JSON envelope."""
    message, code = _error_fields(raw, status)
    if status in (401, 403):
        return LiteratureAuthenticationError(
            message, status_code=status, provider_error=code
        )
    if status == 429:
        return LiteratureRateLimitError(
            message, retry_after_seconds=_retry_after(headers)
        )
    return LiteratureTransportError(message, status_code=status, provider_error=code)


def _error_fields(raw: bytes, status: int) -> tuple[str, str | None]:
    """``(message, error_label)`` from the documented ``{"error",
    "message"}`` envelope — with a plain fallback, because a 404 was
    observed live returning an HTML body instead."""
    fallback = f"the OpenAlex endpoint returned HTTP {status}"
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return fallback, None
    if not isinstance(payload, Mapping):
        return fallback, None
    label = payload.get("error")
    detail = payload.get("message")
    label = label if isinstance(label, str) and label else None
    if isinstance(detail, str) and detail:
        return f"{fallback}: {detail}", label
    if label is not None:
        return f"{fallback}: {label}", label
    return fallback, None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The server's wait, accepted only when finite and non-negative."""
    try:
        seconds = float(headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _lowered(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Header pairs as a plain lower-cased dict — the last stdlib-HTTP-
    shaped object converted before anything leaves the adapter."""
    return {key.lower(): value for key, value in items}
