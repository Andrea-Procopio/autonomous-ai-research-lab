"""arXiv abstract lookup: the one lawful in-repo path to a missing abstract.

Task 5G. OpenAlex sometimes returns a work with no abstract — a
metadata-only source that today can never be extracted, grounded, or
cited. When the index's own record names an arXiv id, the work's
abstract is one bounded fetch away on the work's own listing; this
module performs that fetch, and nothing more.

Standard-library HTTP only, mirroring the OpenAlex adapter: every
vendor-shaped object (the query URL, the Atom XML, ``http.client``
types) stays inside this module. The fetch is best-effort by design —
a missing entry, a transport failure, or malformed XML degrades to
``None``, never to an exception, because access resolution must not be
able to kill a retrieval that already succeeded: an unresolved work
stays honestly metadata-only.

Wire contract, established from the arXiv API documentation
(info.arxiv.org/help/api) and a live response captured on 2026-08-23:
``GET /api/query?id_list=<id>&max_results=1`` returns an Atom feed;
a found work is an ``<entry>`` whose ``<summary>`` holds the abstract;
an unknown id yields either an empty feed or an error entry without a
usable summary. Whitespace inside ``<summary>`` is layout, collapsed
to single spaces here.
"""

from __future__ import annotations

import http.client
import urllib.parse
import xml.etree.ElementTree as ElementTree
from typing import Final

DEFAULT_BASE_URL: Final = "https://export.arxiv.org"
DEFAULT_TIMEOUT_SECONDS: Final = 20.0

_ATOM: Final = "{http://www.w3.org/2005/Atom}"


def fetch_abstract(
    arxiv_id: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """The work's own abstract from its arXiv listing, or ``None``."""
    if not arxiv_id.strip():
        return None
    query = urllib.parse.urlencode(
        {"id_list": arxiv_id.strip(), "max_results": "1"}
    )
    split = urllib.parse.urlsplit(base_url)
    try:
        if split.scheme == "https":
            connection: http.client.HTTPConnection = (
                http.client.HTTPSConnection(
                    split.netloc, timeout=timeout_seconds
                )
            )
        else:
            connection = http.client.HTTPConnection(
                split.netloc, timeout=timeout_seconds
            )
        try:
            connection.request(
                "GET",
                f"/api/query?{query}",
                headers={"User-Agent": "autonomous-research-lab"},
            )
            response = connection.getresponse()
            if response.status != 200:
                return None
            body = response.read()
        finally:
            connection.close()
    except (OSError, http.client.HTTPException):
        return None
    return summary_from(body)


def summary_from(body: bytes) -> str | None:
    """The first entry's summary, whitespace-collapsed — pure parsing,
    so the contract is testable without a network."""
    try:
        feed = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return None
    entry = feed.find(f"{_ATOM}entry")
    if entry is None:
        return None
    title = entry.findtext(f"{_ATOM}title") or ""
    if title.strip() == "Error":
        return None
    summary = entry.findtext(f"{_ATOM}summary")
    if summary is None:
        return None
    collapsed = " ".join(summary.split())
    return collapsed or None
