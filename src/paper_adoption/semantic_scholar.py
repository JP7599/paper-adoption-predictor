"""Talk to Semantic Scholar (https://api.semanticscholar.org): a second, independent catalogue of papers and citations.

Why a second one: Semantic Scholar merges the arXiv version and the journal version of a paper into ONE record and
resolves references to arXiv papers by their arXiv number, so it sees many more citations to preprints than OpenAlex.
With an API key the limit is ONE request per second, cumulative over all endpoints: never run two Semantic Scholar
jobs at the same time. We pace at 1.1 seconds and back off on "429 too many requests".

References: Kinney, R., Anastasiades, C., Authur, R., Beltagy, I., Bragg, J., ... Weld, D. S. (2023). The Semantic Scholar
open data platform. arXiv. https://doi.org/10.48550/arXiv.2301.10140 ; Allen Institute for AI. (2026). Semantic Scholar
Academic Graph API [Application programming interface]. https://api.semanticscholar.org/
"""
from __future__ import annotations

from datetime import date, timedelta

from . import cache
from . import config as C

API = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,externalIds,citationCount,referenceCount,influentialCitationCount,publicationDate,year"
MAX_OFFSET = 9000                     # the citations endpoint cannot page beyond 10,000 citing papers


def _headers() -> dict:
    """The API key from .env, if you have one. Sent as a header, never printed, never part of the cache key."""
    return {"x-api-key": C.S2_API_KEY} if C.S2_API_KEY else {}


def s2_id(paper_id: str) -> str:
    """Our paper_id is an arXiv DOI (10.48550/arxiv.<id>) or a bioRxiv/medRxiv DOI; Semantic Scholar wants ARXIV:<id> or DOI:<doi>."""
    if paper_id.startswith("10.48550/arxiv."):
        return "ARXIV:" + paper_id.split("10.48550/arxiv.")[1]
    return "DOI:" + paper_id


def batch(ids: list[str]) -> dict[str, dict | None]:
    """Look up many papers at once, 500 per request. Returns {our id: record or None when S2 does not know it}."""
    found: dict[str, dict | None] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        answer = cache.post_json(f"{API}/paper/batch", {"ids": chunk}, params={"fields": PAPER_FIELDS}, pace_seconds=1.1, headers=_headers())
        for asked, record in zip(chunk, answer):
            found[asked] = record
    return found


def citing_dates(s2_paper_id: str) -> tuple[list[date], int, bool]:
    """Dates of all papers citing this one. Returns (dates, number of citing papers without a usable date,
    True if the list was cut off at the endpoint's paging limit). Pages of 1,000, one request per page."""
    dates: list[date] = []
    undated = 0
    offset = 0
    truncated = False
    while True:
        answer = cache.get_json(f"{API}/paper/{s2_paper_id}/citations",
                                {"fields": "publicationDate,year", "limit": 1000, "offset": offset}, pace_seconds=1.1, headers=_headers())
        for item in (answer or {}).get("data", []):
            citing = item.get("citingPaper") or {}
            text = citing.get("publicationDate") or (f"{citing['year']}-07-01" if citing.get("year") else None)  # year only -> mid-year
            try:
                dates.append(date.fromisoformat(text[:10]))
            except (TypeError, ValueError):
                undated += 1
        nxt = (answer or {}).get("next")
        if nxt is None:
            break
        if nxt > MAX_OFFSET:
            truncated = True
            break
        offset = nxt
    return dates, undated, truncated


def citations_in_first_year(s2_paper_id: str, posted: date, window_days: int = 365) -> dict:
    """Exact count of citing papers dated within `window_days` after `posted`, according to Semantic Scholar."""
    dates, undated, truncated = citing_dates(s2_paper_id)
    end = posted + timedelta(days=window_days)
    return {"cites_12m_s2": sum(1 for d in dates if posted <= d <= end), "s2_total_seen": len(dates),
            "s2_undated": undated, "s2_truncated": truncated}
