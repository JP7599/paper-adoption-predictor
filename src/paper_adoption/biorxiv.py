"""bioRxiv / medRxiv API (https://api.biorxiv.org): the subject category and version history of a preprint, by DOI.

Free, no key, about one request per second. The endpoint was down for a day during iteration zero and may be again;
callers should treat a failure as "unknown" and move on.

Reference: Cold Spring Harbor Laboratory. (2026). bioRxiv and medRxiv API [Application programming interface]. https://api.biorxiv.org/
"""
from __future__ import annotations

from . import cache

API = "https://api.biorxiv.org/details"


def details(server: str, doi: str) -> dict | None:
    """Category, first-posting date, number of versions and the published DOI (if any) of one preprint."""
    j = cache.get_json(f"{API}/{server}/{doi}/na/json", pace_seconds=1.0, retries=3)
    versions = (j or {}).get("collection") or []
    if not versions:
        return None
    first, last = versions[0], versions[-1]
    published = last.get("published")
    return {"rxiv_category": (first.get("category") or "").strip().lower() or None,
            "rxiv_date_v1": first.get("date"),
            "rxiv_versions": len(versions),
            "rxiv_published_doi": None if published in (None, "", "NA") else published.lower(),
            "rxiv_license": first.get("license")}


PAGE = 30   # records per page of the interval endpoint


def list_interval(server: str, day_from, day_to, category: str | None = None, max_pages: int | None = None) -> list[dict]:
    """Every record posted on `server` between the two dates (one record per VERSION), optionally one category only.
    Pages of 30; we stop when a page is short, when the server's total is reached, or after `max_pages`."""
    rows: list[dict] = []
    cursor = 0
    pages = 0
    while True:
        params = {"category": category.replace(" ", "_")} if category else {}
        j = cache.get_json(f"{API}/{server}/{day_from}/{day_to}/{cursor}/json", params, pace_seconds=1.0, retries=4)
        info = ((j or {}).get("messages") or [{}])[0]
        page = (j or {}).get("collection") or []
        rows += page
        cursor += len(page)
        pages += 1
        total = int(info.get("total") or 0)
        if len(page) < PAGE or cursor >= total or (max_pages and pages >= max_pages):
            return rows


def first_versions(records: list[dict], server: str, field: str) -> list[dict]:
    """Keep version 1 of every preprint and flatten it to our row format (same names as the arXiv rows)."""
    out = []
    for r in records:
        if str(r.get("version")) != "1":
            continue
        published = r.get("published")
        authors = [a.strip() for a in (r.get("authors") or "").split(";") if a.strip()]
        out.append({
            "field": field, "source": server,
            "paper_id": (r.get("doi") or "").lower(),
            "title": " ".join((r.get("title") or "").split()),
            "abstract": " ".join((r.get("abstract") or "").split()),
            "authors": "; ".join(authors), "n_authors": len(authors),
            "posted": r.get("date"),
            "primary_category": None, "categories": None, "comment": "",
            "publisher_doi": None if published in (None, "", "NA") else published.lower(),
            "rxiv_category": (r.get("category") or "").strip().lower() or None,
            "rxiv_license": r.get("license"),
        })
    return out
