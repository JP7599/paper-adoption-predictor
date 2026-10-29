"""The acquisition run for iteration zero: harvest the pilot sample for the five fields, and their outcomes.

    .venv/bin/python -m paper_adoption.acquire

Writes to data/raw/:
    papers_arxiv.parquet      one row per arXiv paper (data_science, physics, economics), as the arXiv API gave it
    papers_openalex.parquet   one row per bioRxiv / medRxiv preprint (biotech, medicine), as OpenAlex gave it
    outcomes_arxiv.parquet    what OpenAlex knows about each arXiv paper (citations, authors' institutions, topic)
    exact_cites.parquet       the exact 12-month citation count for a subsample of papers in every field
Every web answer is cached, so re-running after an interruption continues where it stopped.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm

from . import arxiv, openalex
from . import config as C


def months(month_list: list[tuple[int, int]] | None = None) -> list[date]:
    """The first day of every month in the study window: 2022-01-01, 2022-02-01, ... 2024-12-01 (or of `month_list`)."""
    if month_list is None:
        month_list = [(y, m) for y in C.YEARS for m in range(1, 13)]
    return [date(y, m, 1) for y, m in month_list]


def last_day_of(month: date) -> date:
    return (month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


# ------------------------------------------------------------------------------------------ arXiv fields ----
def harvest_arxiv(field: str, spec: dict, month_list=None, per_field: int | None = None) -> pd.DataFrame:
    """Two snapshots per month: the first papers submitted on or after the 1st, and on or after the 15th."""
    per_field = per_field or C.PILOT_PER_FIELD
    n_windows = len(months(month_list)) * len(C.ARXIV_ANCHOR_DAYS)   # 36 months x 2 anchors = 72 windows
    per_window = math.ceil(per_field / n_windows)                     # 1000 / 72 -> 14 papers per window
    rows = []
    for month in tqdm(months(month_list), desc=f"arXiv  {field:13s}"):
        for day in C.ARXIV_ANCHOR_DAYS:
            start = month.replace(day=day)
            end = start + timedelta(days=C.ARXIV_WINDOW_DAYS - 1)
            got = arxiv.search(spec["query_categories"], start, end, max_results=2 * per_window)  # ask for extra...
            got = [r for r in got if r["primary_category"].startswith(tuple(spec["primary_prefixes"]))]  # ...keep the right primaries
            rows += got[:per_window]
    df = pd.DataFrame(rows).drop_duplicates("arxiv_id")
    df.insert(0, "field", field)
    df.insert(1, "source", "arxiv")
    return df


def harvest_arxiv_daily(field: str, spec: dict, start: date, end: date, per_day: int, request_n: int | None = None) -> pd.DataFrame:
    """The full-size harvest: the first `per_day` papers (with the right primary category) submitted on every single
    day from `start` to `end`. One request per day; `request_n` asks for extra so the primary-category filter can drop some."""
    request_n = request_n or max(per_day * 2, 20)
    rows = []
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    skipped = []
    for day in tqdm(days, desc=f"arXiv  {field:13s}"):
        try:
            got = arxiv.search(spec["query_categories"], day, day, max_results=min(request_n, 2000))
        except RuntimeError as e:                                  # a day that fails for good is skipped, not fatal
            skipped.append(day)
            print(f"  skipped {day}: {e}", file=sys.stderr, flush=True)
            continue
        got = [r for r in got if r["primary_category"].startswith(tuple(spec["primary_prefixes"]))]
        rows += got[:per_day]
    if skipped:
        print(f"  {field}: {len(skipped)} days skipped; re-run to fetch them (the rest is cached)", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows).drop_duplicates("arxiv_id")
    df.insert(0, "field", field)
    df.insert(1, "source", "arxiv")
    df["paper_id"] = "10.48550/arxiv." + df["arxiv_id"]
    return df


def outcomes_for_papers(papers: pd.DataFrame) -> pd.DataFrame:
    """Like outcomes_for_arxiv, for any table with `paper_id` (a DOI) and an optional `publisher_doi`:
    arXiv papers (10.48550/arxiv.<id>) and bioRxiv/medRxiv preprints (10.1101/...) alike."""
    preprint_dois = papers["paper_id"].str.lower().tolist()
    publisher_dois = papers["publisher_doi"].dropna().tolist()
    found = openalex.works_by_doi(preprint_dois + publisher_dois)
    rows = []
    for paper, pre_doi in zip(papers.itertuples(index=False), preprint_dois):
        publisher_doi = paper.publisher_doi.lower() if isinstance(paper.publisher_doi, str) else None
        records = list({w["id"]: w for w in (found.get(pre_doi), found.get(publisher_doi)) if w}.values())
        if not records:
            rows.append({"paper_id": paper.paper_id, "matched": 0, "later_published": int(publisher_doi is not None)})
            continue
        flats = [openalex.flatten_work(w) for w in records]
        main = dict(max(flats, key=lambda f: f["cited_total"]))
        main["openalex_ids"] = "|".join(f["openalex_id"] for f in flats)
        main["n_records"] = len(flats)
        main["cited_total"] = sum(f["cited_total"] for f in flats)
        main["counts_by_year"] = _sum_counts_by_year([f["counts_by_year"] for f in flats])
        for key in ("n_institutions", "n_countries", "n_references", "oa_n_authors", "n_locations"):
            main[key] = max(f[key] for f in flats)
        main["later_published"] = int(publisher_doi is not None or any(f["openalex_type"] != "preprint" for f in flats))
        rows.append({"paper_id": paper.paper_id, "matched": 1, **main})
    return _jsonify_counts(pd.DataFrame(rows))


def outcomes_for_arxiv(papers: pd.DataFrame) -> pd.DataFrame:
    """Find every arXiv paper in OpenAlex through its DOI(s): the arXiv DOI that every paper has (10.48550/arXiv.<id>)
    and, when the authors registered one, the publisher's DOI. OpenAlex may hold them as two separate records of the
    same paper; then we add their citations up and remember both ids."""
    arxiv_dois = ("10.48550/arxiv." + papers["arxiv_id"]).str.lower().tolist()
    publisher_dois = papers["publisher_doi"].dropna().tolist()
    found = openalex.works_by_doi(arxiv_dois + publisher_dois)
    rows = []
    for paper, arxiv_doi in zip(papers.itertuples(index=False), arxiv_dois):
        publisher_doi = paper.publisher_doi.lower() if isinstance(paper.publisher_doi, str) else None  # missing -> None
        records = [found.get(arxiv_doi), found.get(publisher_doi)]
        records = list({w["id"]: w for w in records if w}.values())        # drop misses and duplicates
        if not records:
            rows.append({"arxiv_id": paper.arxiv_id, "matched": 0, "later_published": int(publisher_doi is not None)})
            continue
        flats = [openalex.flatten_work(w) for w in records]
        main = dict(max(flats, key=lambda f: f["cited_total"]))            # the better-known record gives the metadata
        main["openalex_ids"] = "|".join(f["openalex_id"] for f in flats)
        main["n_records"] = len(flats)
        main["cited_total"] = sum(f["cited_total"] for f in flats)
        main["counts_by_year"] = _sum_counts_by_year([f["counts_by_year"] for f in flats])
        for key in ("n_institutions", "n_countries", "n_references", "oa_n_authors", "n_locations"):
            main[key] = max(f[key] for f in flats)                          # the preprint record often lacks institutions
        main["later_published"] = int(publisher_doi is not None or any(f["openalex_type"] != "preprint" for f in flats))
        rows.append({"arxiv_id": paper.arxiv_id, "matched": 1, **main})
    return _jsonify_counts(pd.DataFrame(rows))


def _sum_counts_by_year(lists: list[list[dict]]) -> list[dict]:
    total: dict[int, int] = {}
    for counts in lists:
        for c in counts:
            total[c["year"]] = total.get(c["year"], 0) + c["cited_by_count"]
    return [{"year": y, "cited_by_count": n} for y, n in sorted(total.items())]


def _jsonify_counts(df: pd.DataFrame) -> pd.DataFrame:
    """A list of dictionaries does not fit a table cell; store it as a JSON string."""
    if "counts_by_year" in df:
        df["counts_by_year"] = df["counts_by_year"].apply(lambda v: json.dumps(v) if isinstance(v, list) else None)
    return df


# ------------------------------------------------------------------------------- bioRxiv / medRxiv fields ----
def harvest_openalex(field: str, spec: dict, month_list=None, per_field: int | None = None) -> pd.DataFrame:
    """A reproducible random sample per month of the preprints the server hosted, with outcomes included."""
    per_field = per_field or C.PILOT_PER_FIELD
    per_month = math.ceil(per_field / len(months(month_list)))        # 1000 / 36 -> 28 papers per month
    rows = []
    for i, month in enumerate(tqdm(months(month_list), desc=f"OpenAlex {spec['label']:11s}")):
        works = openalex.sample_from_source(spec["openalex_source_id"], month, last_day_of(month), per_month, seed=1000 + i)
        for w in works:
            flat = openalex.flatten_work(w)
            rows.append({
                **flat,
                "title": flat["openalex_title"], "abstract": flat["abstract_openalex"],
                "posted": flat["openalex_date"], "n_authors": flat["oa_n_authors"],
                "openalex_ids": flat["openalex_id"], "n_records": 1, "matched": 1,
                "later_published": int(flat["openalex_type"] != "preprint"),
            })
    df = pd.DataFrame(rows).drop_duplicates("openalex_id")
    df.insert(0, "field", field)
    df.insert(1, "source", spec["label"].lower())
    return _jsonify_counts(df)


# --------------------------------------------------------------------------------- exact 12-month counts ----
def exact_cites_subsample(papers: pd.DataFrame, per_field: int | None = None) -> pd.DataFrame:
    """The exact 12-month count for EXACT_CITES_PER_FIELD random matched papers per field (one request each).
    The notebook compares these with the cheap estimate that every paper has."""
    per_field = C.EXACT_CITES_PER_FIELD if per_field is None else per_field
    rows = []
    for field, group in papers[papers["matched"] == 1].groupby("field"):
        sub = group.sample(n=min(per_field, len(group)), random_state=0)
        for p in tqdm(sub.itertuples(index=False), total=len(sub), desc=f"exact  {field:13s}"):
            n = openalex.citations_in_first_year(p.openalex_ids.split("|"), date.fromisoformat(p.posted))
            rows.append({"paper_id": p.paper_id, "cites_12m_exact": n})
    return pd.DataFrame(rows, columns=["paper_id", "cites_12m_exact"])


def main(month_list=None, per_field: int | None = None, exact_per_field: int | None = None, prefix: str = "") -> None:
    """The whole run. With the defaults it builds the 2022-2024 pilot; iteration 3 calls it again with the hold-out
    months, bigger sizes and the file prefix "holdout_"."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)

    oa = pd.concat([harvest_openalex(f, s, month_list, per_field) for f, s in C.FIELDS.items() if s["source"] == "openalex"], ignore_index=True)
    oa.to_parquet(C.RAW_DIR / f"{prefix}papers_openalex.parquet", index=False)
    print(f"bioRxiv/medRxiv preprints: {len(oa)}  ->  {C.RAW_DIR / (prefix + 'papers_openalex.parquet')}", flush=True)

    arx = pd.concat([harvest_arxiv(f, s, month_list, per_field) for f, s in C.FIELDS.items() if s["source"] == "arxiv"], ignore_index=True)
    arx.to_parquet(C.RAW_DIR / f"{prefix}papers_arxiv.parquet", index=False)
    print(f"arXiv papers: {len(arx)}  ->  {C.RAW_DIR / (prefix + 'papers_arxiv.parquet')}", flush=True)

    out = outcomes_for_arxiv(arx)
    out.to_parquet(C.RAW_DIR / f"{prefix}outcomes_arxiv.parquet", index=False)
    print(f"arXiv papers found in OpenAlex: {int(out['matched'].sum())} of {len(out)}", flush=True)

    # one light combined view, only to choose the subsample for the exact counts
    arx_view = arx[["field", "arxiv_id", "posted"]].merge(out[["arxiv_id", "matched", "openalex_ids"]], on="arxiv_id", how="left")
    arx_view["paper_id"] = "10.48550/arxiv." + arx_view["arxiv_id"]
    oa_view = oa[["field", "posted", "matched", "openalex_ids"]].assign(paper_id=oa["openalex_doi"].fillna(oa["openalex_id"]))
    both = pd.concat([arx_view, oa_view], ignore_index=True)
    both["matched"] = both["matched"].fillna(0).astype(int)
    exact = exact_cites_subsample(both, exact_per_field)
    exact.to_parquet(C.RAW_DIR / f"{prefix}exact_cites.parquet", index=False)
    print(f"exact 12-month counts fetched for {len(exact)} papers  ->  {C.RAW_DIR / (prefix + 'exact_cites.parquet')}", flush=True)


if __name__ == "__main__":
    main()
