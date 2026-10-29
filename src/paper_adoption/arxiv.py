"""Fetch paper metadata from the arXiv API (https://info.arxiv.org/help/api/).

Free, no key. arXiv asks for at most one request every 3 seconds, which `cache.get_text` enforces.
The API answers in Atom XML; the `feedparser` library turns that into Python objects for us.

Reference: Cornell University. (2026). arXiv API [Application programming interface]. arXiv. https://info.arxiv.org/help/api/
"""
from __future__ import annotations

import re
from datetime import date

import feedparser

from . import cache

API = "https://export.arxiv.org/api/query"


def search(categories: list[str], day_from: date, day_to: date, max_results: int) -> list[dict]:
    """Papers whose first version was submitted between `day_from` and `day_to` (inclusive),
    listed in at least one of `categories`, oldest first, at most `max_results` of them."""
    cats = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cats}) AND submittedDate:[{day_from:%Y%m%d}0000 TO {day_to:%Y%m%d}2359]"
    text = cache.get_text(
        API,
        {"search_query": query, "start": 0, "max_results": max_results,
         "sortBy": "submittedDate", "sortOrder": "ascending"},
        pace_seconds=4.0, retries=10,
        # the API sometimes answers with an empty feed although papers exist; such an answer must not be cached.
        # An answer is genuine if it has entries, or if it honestly says there are 0 results.
        valid=lambda t: "<entry>" in t or re.search(r"totalResults[^>]*>0<", t) is not None,
    )
    feed = feedparser.parse(text)
    return [_to_row(entry) for entry in feed.entries]


def _to_row(e) -> dict:
    """Keep the handful of fields we need, with plain names, one flat dictionary per paper."""
    arxiv_id = re.sub(r"v\d+$", "", e.id.split("/abs/")[-1])      # ".../abs/2203.02155v1" -> "2203.02155"
    doi = (e.get("arxiv_doi") or "").strip().lower() or None       # present only if the authors added it, later
    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(e.title.split()),                          # the feed wraps titles over several lines
        "abstract": " ".join(e.summary.split()),
        "authors": "; ".join(a.name for a in e.authors),
        "n_authors": len(e.authors),
        "posted": e.published[:10],                                  # date of the first version, "2022-03-04"
        "primary_category": e.arxiv_primary_category["term"],
        "categories": " ".join(t.term for t in e.tags),              # primary + cross-lists
        "comment": " ".join((e.get("arxiv_comment") or "").split()),
        "publisher_doi": doi,
    }
