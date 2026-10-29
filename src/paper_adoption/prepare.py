"""Turn the raw downloads into ONE tidy table for the notebook: data/processed/papers.csv, one row per paper.

    .venv/bin/python -m paper_adoption.prepare

Everything technical (joining the arXiv papers to their OpenAlex records, reconstructing abstracts, estimating the
12-month citation count, spotting code links) happens here, once, so the notebooks only have to read a table.

Iteration 1 added enrichment files in data/raw/ (Semantic Scholar, Hugging Face, bioRxiv categories; see enrich.py).
If they exist they are merged in, and the cohort-relative labels are computed; if not, those columns are simply absent.
"""
from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from .textclean import clean_series
from . import openalex

CODE_LINK = re.compile(r"(?:github|gitlab|bitbucket)\.(?:com|org)/[\w.-]+/[\w.-]+", re.IGNORECASE)   # (?:...) = group without capturing

COLUMNS = [  # the order in which the columns appear in the CSV: identity, text, publication-day features, outcomes
    "paper_id", "field", "source", "posted", "year", "month", "title", "abstract",
    "n_authors", "title_words", "abstract_words", "has_code_link", "primary_category", "n_categories",
    "matched", "openalex_ids", "n_records", "n_references", "n_institutions", "n_countries",
    "first_author_country", "first_author_institution_type", "topic", "subfield", "oa_field", "language",
    "later_published", "is_retracted", "cited_total", "cites_12m", "cites_12m_exact", "counts_by_year",
]
ENRICHMENT = {  # file in data/raw -> columns taken from it (all optional)
    "s2_papers.parquet": ["s2_found", "s2_via", "s2_paper_id", "s2_cited_total", "s2_references", "s2_influential"],
    "s2_cites12m.parquet": ["cites_12m_s2", "s2_undated", "s2_truncated"],
    "hf_papers.parquet": ["hf_page", "hf_repo", "hf_stars", "hf_upvotes", "hf_models", "hf_datasets"],
    "rxiv_details.parquet": ["rxiv_category", "rxiv_versions", "rxiv_published_doi"],
    "author_prestige.parquet": ["n_author_ids", "first_author_prior_works", "first_author_prior_cites",
                                "last_author_prior_works", "last_author_prior_cites", "max_prior_cites"],
}


def top10_within(groups: pd.Series, values: pd.Series) -> pd.Series:
    """1 if the value is at or above the 90th percentile of its group, 0 below, missing where the value is missing."""
    threshold = values.groupby(groups).transform(lambda s: s.quantile(0.90))
    label = (values >= threshold).astype("float")
    label[values.isna()] = np.nan
    return label.astype("Int64")


def main(prefix: str = "", out_name: str = "papers.csv") -> None:
    """Build the tidy table. prefix="" gives the 2022-2024 pilot (papers.csv); prefix="holdout_" the sealed 2025 set."""
    raw = C.RAW_DIR
    if (raw / f"{prefix}papers_harvest.parquet").exists():
        # ---- the full harvest: one table of harvested papers (arXiv + bioRxiv + medRxiv) and one of OpenAlex outcomes
        papers = pd.read_parquet(raw / f"{prefix}papers_harvest.parquet")
        if (raw / f"{prefix}outcomes.parquet").exists():
            papers = papers.merge(pd.read_parquet(raw / f"{prefix}outcomes.parquet"), on="paper_id", how="left")
        else:
            papers["matched"] = 0
        papers["matched"] = papers["matched"].fillna(0).astype(int)
        for col in ("openalex_ids", "n_records", "n_references", "n_institutions", "n_countries", "first_author_country",
                    "first_author_institution_type", "topic", "subfield", "oa_field", "language", "is_retracted",
                    "cited_total", "counts_by_year", "later_published"):
            if col not in papers:
                papers[col] = np.nan
        papers["later_published"] = papers["later_published"].fillna(papers["publisher_doi"].notna().astype(int))
    else:
        # ---- the pilot: arXiv papers + what OpenAlex knows about them ------------------------------------------
        arx = pd.read_parquet(raw / f"{prefix}papers_arxiv.parquet")
        out = pd.read_parquet(raw / f"{prefix}outcomes_arxiv.parquet")
        arx = arx.merge(out, on="arxiv_id", how="left")
        arx["paper_id"] = "10.48550/arxiv." + arx["arxiv_id"]
        arx["matched"] = arx["matched"].fillna(0).astype(int)

        # ---- bioRxiv / medRxiv preprints (outcomes already included) ----------------------------------------
        oa = pd.read_parquet(raw / f"{prefix}papers_openalex.parquet")
        oa["paper_id"] = oa["openalex_doi"].fillna(oa["openalex_id"])

        papers = pd.concat([arx, oa], ignore_index=True)

    # ---- publication-day features: things knowable the day the paper appeared ----------------------------------
    papers["abstract"] = clean_series(papers["abstract"])      # strip the figure markup some bioRxiv abstracts carry (textclean.py)
    text = papers["title"].fillna("") + " " + papers["abstract"].fillna("") + " " + papers.get("comment", pd.Series("", index=papers.index)).fillna("")
    papers["has_code_link"] = text.str.contains(CODE_LINK).astype(int)
    papers["title_words"] = papers["title"].fillna("").str.split().str.len()
    papers["abstract_words"] = papers["abstract"].fillna("").str.split().str.len()
    papers["n_categories"] = papers["categories"].str.split().str.len()      # arXiv only; empty for bioRxiv/medRxiv
    posted = pd.to_datetime(papers["posted"])
    papers["posted"] = posted.dt.strftime("%Y-%m-%d")
    papers["year"] = posted.dt.year
    papers["month"] = posted.dt.strftime("%Y-%m")

    # ---- outcomes -------------------------------------------------------------------------------------------
    papers["cites_12m"] = [
        openalex.approx_citations_in_first_year(json.loads(c), d.date()) if isinstance(c, str) else np.nan
        for c, d in zip(papers["counts_by_year"], posted)
    ]
    if (raw / f"{prefix}exact_cites.parquet").exists():
        papers = papers.merge(pd.read_parquet(raw / f"{prefix}exact_cites.parquet"), on="paper_id", how="left")
    else:
        papers["cites_12m_exact"] = np.nan
    papers["later_published"] = papers["later_published"].fillna(0).astype(int)
    for col in ("n_records", "n_references", "n_institutions", "n_countries", "cited_total"):
        papers[col] = papers[col].astype("Int64")                             # whole numbers that may be missing

    keep = COLUMNS + [c for c in ("rxiv_category", "rxiv_license") if c in papers and c not in COLUMNS]
    papers = papers[keep].sort_values(["field", "posted"]).reset_index(drop=True)

    # ---- iteration 1: merge the enrichment files that exist ---------------------------------------------------
    for filename, cols in ENRICHMENT.items():
        path = raw / f"{prefix}{filename}"
        if path.exists():
            extra = pd.read_parquet(path)
            take = ["paper_id"] + [c for c in cols if c in extra and c not in papers]      # never overwrite a harvested column
            papers = papers.merge(extra[take], on="paper_id", how="left")
    for cols in ENRICHMENT.values():                           # columns of a missing enrichment file exist anyway, empty
        for col in cols:
            if col not in papers:
                papers[col] = np.nan
    if "rxiv_category" in papers:
        papers["in_biotech_categories"] = papers["rxiv_category"].isin(C.BIOTECH_CATEGORIES).astype(int)
    if "rxiv_published_doi" in papers:                       # bioRxiv knows the journal version: a better "later published"
        papers["later_published"] = (papers["later_published"].astype(bool) | papers["rxiv_published_doi"].notna()).astype(int)

    # ---- labels: top 10 % within field (iteration zero) and within field x half-year cohort (iteration 1) --------
    month_num = pd.to_datetime(papers["posted"]).dt.month
    papers["half_year"] = papers["year"].astype(str) + np.where(month_num <= 6, " H1", " H2")
    papers["cohort"] = papers["field"] + " " + papers["half_year"]
    papers["top10_oa_field"] = top10_within(papers["field"], papers["cites_12m"])
    papers["top10_oa_cohort"] = top10_within(papers["cohort"], papers["cites_12m"])
    papers["top10_s2_cohort"] = top10_within(papers["cohort"], papers["cites_12m_s2"])
    papers["top10_s2total_cohort"] = top10_within(papers["cohort"], papers["s2_cited_total"])   # within a cohort all papers had the same time to be cited
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    papers.to_csv(C.PROCESSED_DIR / out_name, index=False)
    print(f"{len(papers)} papers  ->  {C.PROCESSED_DIR / out_name}")
    summary = {"papers": ("paper_id", "size"), "matched": ("matched", "mean"), "exact": ("cites_12m_exact", "count")}
    for col in ("s2_found", "cites_12m_s2", "hf_repo", "rxiv_category"):
        if col in papers:
            summary[col] = (col, "count")
    print(papers.groupby("field").agg(**summary).round(3))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
