"""Iteration 3: leak-safe "prestige" features from the authors' track record BEFORE the paper was posted.

    .venv/bin/python -m paper_adoption.authors

Idea: how established were the first and the last author when the paper appeared? OpenAlex keeps, per author, how
many works they published and how many citations they received in each calendar year. Summing only the years
BEFORE the posting year gives a number that was knowable on the posting day. (Citations received in the posting year
itself are left out entirely, because we cannot split them into before and after the posting day: conservative.)

Author ids come from the OpenAlex work records that iteration zero already cached (no new requests); the author
look-ups themselves (/authors/<id>) are free in OpenAlex's budget. Writes data/raw/author_prestige.parquet.

Reference: OurResearch. (2026). OpenAlex API [Application programming interface]. https://docs.openalex.org/ (author histories).
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
from tqdm import tqdm

from . import cache
from . import config as C

AUTHOR_API = "https://api.openalex.org/authors"
AUTHOR_FIELDS = "id,display_name,works_count,cited_by_count,counts_by_year"


def authorships_from_cache() -> dict[str, list[dict]]:
    """Read every cached OpenAlex answer and collect {work id: authorships} for all works it contains."""
    found: dict[str, list[dict]] = {}
    for path in C.CACHE_DIR.rglob("*.txt"):
        text = path.read_text()
        if not text.startswith('{"meta"'):
            continue
        for work in json.loads(text).get("results", []):
            if work.get("authorships") is not None:
                wid = work["id"].split("/")[-1]
                if len(work["authorships"]) >= len(found.get(wid, [])):   # keep the most complete record
                    found[wid] = work["authorships"]
    return found


def author_record(author_id: str) -> dict | None:
    return cache.get_json(f"{AUTHOR_API}/{author_id}", {"select": AUTHOR_FIELDS}, pace_seconds=0.2, not_found_ok=True)


def prior_counts(record: dict | None, posted: date) -> tuple[float | None, float | None]:
    """(works before the posting year, citations received before the posting year) for one author."""
    if not record:
        return None, None
    before = [c for c in (record.get("counts_by_year") or []) if c["year"] < posted.year]   # the history can be null
    return sum(c.get("works_count", 0) for c in before), sum(c.get("cited_by_count", 0) for c in before)


def main(prefix: str = "", table: str = "papers.csv") -> None:
    papers = pd.read_csv(C.PROCESSED_DIR / table, usecols=["paper_id", "posted", "openalex_ids", "matched"])
    papers = papers[papers["matched"] == 1]
    by_work = authorships_from_cache()
    print(f"author lists found in the cache for {len(by_work)} work records", flush=True)

    out_path = C.RAW_DIR / f"{prefix}author_prestige.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["paper_id"])   # resume after a crash
    papers = papers[~papers["paper_id"].isin(done["paper_id"])]
    rows = done.to_dict("records")
    for i, p in enumerate(tqdm(papers.itertuples(index=False), total=len(papers), desc="author prestige")):
        records = [by_work.get(w) for w in str(p.openalex_ids).split("|")]
        authorships = max((r for r in records if r), key=len, default=[])
        ids = [((a.get("author") or {}).get("id") or "").split("/")[-1] for a in authorships]
        ids = [i for i in ids if i]
        posted = date.fromisoformat(p.posted)
        first = author_record(ids[0]) if ids else None
        last = author_record(ids[-1]) if len(ids) > 1 else first
        fw, fc = prior_counts(first, posted)
        lw, lc = prior_counts(last, posted)
        rows.append({"paper_id": p.paper_id, "n_author_ids": len(ids), "n_authorships": len(authorships),
                     "first_author_prior_works": fw, "first_author_prior_cites": fc,
                     "last_author_prior_works": lw, "last_author_prior_cites": lc,
                     "max_prior_cites": max([x for x in (fc, lc) if x is not None], default=None)})
        if (i + 1) % 500 == 0:                                                   # save progress every 500 papers
            pd.DataFrame(rows).to_parquet(out_path, index=False)
    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"prestige features for {len(out)} papers ({int((out['n_author_ids'] > 0).sum())} with at least one author id)"
          f"  ->  {prefix}author_prestige.parquet", flush=True)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(prefix=args[0] if args else "", table=args[1] if len(args) > 1 else "papers.csv")
