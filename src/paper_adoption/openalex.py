"""Talk to OpenAlex (https://openalex.org): the free, open catalogue of papers, authors and citations.

We use it in two ways:
  * as the OUTCOME source for arXiv papers: look works up by DOI and read their citation counts;
  * as the PAPER source for bioRxiv and medRxiv: draw a random sample of the preprints each server hosted in a month.
Budget: anonymous use is metered (about 1,000 list requests per day when this was written; a free API key gives
about ten times that). Every answer is cached on disk, so a re-run costs nothing.

References: Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues,
institutions, and concepts. arXiv. https://doi.org/10.48550/arXiv.2205.01833 ; OurResearch. (2026). OpenAlex API
[Application programming interface]. https://docs.openalex.org/
"""
from __future__ import annotations

from datetime import date, timedelta

from . import cache
from . import config as C

API = "https://api.openalex.org/works"

# the parts of a work we ask for (asking for less makes answers smaller and faster)
WORK_FIELDS = ",".join([
    "id", "doi", "title", "publication_date", "type", "cited_by_count", "counts_by_year",
    "referenced_works_count", "authorships", "primary_location", "primary_topic",
    "abstract_inverted_index", "locations_count", "indexed_in", "is_retracted", "language",
])


def _auth() -> dict:
    """Identify ourselves to OpenAlex if you put an e-mail or key in .env. Sent, but never printed or cached."""
    extra = {}
    if C.OPENALEX_MAILTO:
        extra["mailto"] = C.OPENALEX_MAILTO
    if C.OPENALEX_API_KEY:
        extra["api_key"] = C.OPENALEX_API_KEY
    return extra


def _get(params: dict) -> dict:
    return cache.get_json(API, params, extra=_auth(), pace_seconds=0.15)


def works_by_doi(dois: list[str]) -> dict[str, dict]:
    """Look many DOIs up at once, 50 per request. Returns {doi (lower case): work} for the ones OpenAlex knows."""
    clean = sorted({d.lower().strip() for d in dois if d and "," not in d and "|" not in d})  # , and | would break the filter
    found: dict[str, dict] = {}
    for i in range(0, len(clean), 50):
        chunk = clean[i:i + 50]
        answer = _get({"filter": "doi:" + "|".join(chunk), "per-page": 50, "select": WORK_FIELDS})
        for w in answer["results"]:
            found[w["doi"].replace("https://doi.org/", "").lower()] = w
    return found


def sample_from_source(source_id: str, day_from: date, day_to: date, n: int, seed: int) -> list[dict]:
    """A reproducible random sample of `n` works hosted by a source (bioRxiv, medRxiv) and dated inside the window."""
    answer = _get({
        "filter": f"locations.source.id:{source_id},from_publication_date:{day_from},to_publication_date:{day_to}",
        "sample": n, "seed": seed, "per-page": n, "select": WORK_FIELDS,
    })
    return answer["results"]


def citations_in_first_year(work_ids: list[str], posted: date) -> int:
    """EXACT number of distinct works that cite any of `work_ids` (preprint and published version together) and are
    dated within OUTCOME_WINDOW_DAYS after `posted`. One request per paper, so we do this for a subsample only."""
    end = posted + timedelta(days=C.OUTCOME_WINDOW_DAYS)
    ids = "|".join(w.split("/")[-1] for w in work_ids)
    answer = _get({"filter": f"cites:{ids},from_publication_date:{posted},to_publication_date:{end}",
                   "per-page": 1, "select": "id"})
    return answer["meta"]["count"]


def approx_citations_in_first_year(counts_by_year: list[dict], posted: date, as_of: date | None = None) -> float:
    """CHEAP estimate of the same number from OpenAlex's citations-per-calendar-year, which come free with every work.
    Every citation dated in the posting year is counted (it necessarily came after the posting), plus the share of the
    next year's citations that falls inside the 12-month window, pro rata. Posted in March: all of this year's
    citations + 2/12 of next year's. If the next year is the current year (so only part of it has happened), the
    share is taken of the months observed so far instead of twelve."""
    as_of = as_of or date.today()
    by_year = {c["year"]: c["cited_by_count"] for c in counts_by_year}
    months_of_next_year_in_window = posted.month - 1
    months_observed = 12 if posted.year + 1 < as_of.year else max(1, min(12, as_of.month))   # the current year is incomplete
    share = min(1.0, months_of_next_year_in_window / months_observed)
    return by_year.get(posted.year, 0) + by_year.get(posted.year + 1, 0) * share


def abstract_from_inverted_index(inv: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; this puts the words back in order."""
    if not inv:
        return None
    positions = sorted((p, word) for word, ps in inv.items() for p in ps)
    return " ".join(word for _, word in positions)


def flatten_work(w: dict) -> dict:
    """The handful of fields we keep from a work, flat and plainly named."""
    authorships = w.get("authorships") or []
    institutions = [i for a in authorships for i in (a.get("institutions") or [])]
    countries = {c for a in authorships for c in (a.get("countries") or [])}
    first = authorships[0] if authorships else {}
    first_inst = (first.get("institutions") or [{}])[0]
    topic = w.get("primary_topic") or {}
    return {
        "openalex_id": w["id"].split("/")[-1],
        "openalex_doi": (w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
        "openalex_title": w.get("title"),
        "openalex_date": w.get("publication_date"),
        "openalex_type": w.get("type"),
        "cited_total": w.get("cited_by_count") or 0,
        "counts_by_year": w.get("counts_by_year") or [],
        "n_references": w.get("referenced_works_count") or 0,
        "oa_n_authors": len(authorships),
        "n_institutions": len({i.get("id") for i in institutions if i.get("id")}),
        "n_countries": len(countries),
        "first_author_country": (first.get("countries") or [None])[0],
        "first_author_institution_type": first_inst.get("type"),
        "topic": topic.get("display_name"),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "oa_field": (topic.get("field") or {}).get("display_name"),
        "n_locations": w.get("locations_count") or 0,
        "is_retracted": bool(w.get("is_retracted")),
        "language": w.get("language"),
        "abstract_openalex": abstract_from_inverted_index(w.get("abstract_inverted_index")),
    }
