"""Iteration 1 acquisition: add a second citation source, code links and subject categories to the pilot papers.

    .venv/bin/python -m paper_adoption.enrich s2        # Semantic Scholar: totals for all papers, then exact 12-month counts
    .venv/bin/python -m paper_adoption.enrich s2total   #   ... only the totals (fast, 11 requests)
    .venv/bin/python -m paper_adoption.enrich s2window  #   ... only the exact 12-month counts (slow, one request per paper)
    .venv/bin/python -m paper_adoption.enrich hf        # Hugging Face: GitHub repo + stars for the data-science papers
    .venv/bin/python -m paper_adoption.enrich rxiv      # bioRxiv / medRxiv: subject category and versions, by DOI
    .venv/bin/python -m paper_adoption.enrich all

Each part reads data/processed/papers.csv (the table prepare.py built) and writes one parquet file to data/raw/.
Everything is cached, so an interrupted run continues where it stopped; the slow parts save progress as they go.
Afterwards run `python -m paper_adoption.prepare` again: it merges whatever enrichment files exist into papers.csv.
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import biorxiv, huggingface, semantic_scholar
from . import config as C


def load_papers(table: str = "papers.csv") -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED_DIR / table, usecols=["paper_id", "field", "source", "posted"])


def arxiv_id_of(paper_id: str) -> str:
    return paper_id.split("10.48550/arxiv.")[1]


# ------------------------------------------------------------------------------------------ Semantic Scholar ----
def run_s2(papers: pd.DataFrame, prefix: str = "") -> None:
    run_s2_total(papers, prefix)
    run_s2_window(papers, prefix)


def run_s2_total(papers: pd.DataFrame, prefix: str = "") -> None:
    """Totals for all papers: 500 per request, so about 11 requests."""
    ids = [semantic_scholar.s2_id(p) for p in papers["paper_id"]]
    found = semantic_scholar.batch(ids)                                        # 500 per request -> 11 requests
    rows = []
    for paper_id, asked in zip(papers["paper_id"], ids):
        rec = found.get(asked)
        rows.append({"paper_id": paper_id, "s2_found": int(rec is not None),
                     "s2_paper_id": rec and rec.get("paperId"),
                     "s2_cited_total": rec and rec.get("citationCount"),
                     "s2_references": rec and rec.get("referenceCount"),
                     "s2_influential": rec and rec.get("influentialCitationCount"),
                     "s2_publication_date": rec and rec.get("publicationDate")})
    totals = pd.DataFrame(rows)
    totals["s2_via"] = np.where(totals["s2_found"] == 1, "preprint id", None)

    # second chance for bioRxiv/medRxiv preprints S2 does not know under their own DOI: try the journal version's DOI
    rxiv_path = C.RAW_DIR / f"{prefix}rxiv_details.parquet"
    if rxiv_path.exists():
        published = pd.read_parquet(rxiv_path).dropna(subset=["rxiv_published_doi"]).set_index("paper_id")["rxiv_published_doi"]
        retry = totals[(totals["s2_found"] == 0) & totals["paper_id"].isin(published.index)]
        if len(retry):
            found2 = semantic_scholar.batch(["DOI:" + published[p] for p in retry["paper_id"]])
            for idx, paper_id in zip(retry.index, retry["paper_id"]):
                rec = found2.get("DOI:" + published[paper_id])
                if rec:
                    totals.loc[idx, ["s2_found", "s2_paper_id", "s2_cited_total", "s2_references", "s2_influential", "s2_publication_date", "s2_via"]] = \
                        [1, rec.get("paperId"), rec.get("citationCount"), rec.get("referenceCount"), rec.get("influentialCitationCount"), rec.get("publicationDate"), "published doi"]
            print(f"second chance through the journal DOI: {int((totals['s2_via'] == 'published doi').sum())} of {len(retry)} recovered", flush=True)

    totals.to_parquet(C.RAW_DIR / f"{prefix}s2_papers.parquet", index=False)
    print(f"Semantic Scholar knows {int(totals['s2_found'].sum())} of {len(totals)} papers  ->  s2_papers.parquet", flush=True)


def run_s2_window(papers: pd.DataFrame, prefix: str = "") -> None:
    """Exact 12-month counts: one request per paper (more for papers with over 1,000 citations). Slow, so the
    papers are visited in a seeded random order: whatever part is finished is a fair sample of all cohorts."""
    totals = pd.read_parquet(C.RAW_DIR / f"{prefix}s2_papers.parquet")
    out_path = C.RAW_DIR / f"{prefix}s2_cites12m.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["paper_id"])
    todo = totals[(totals["s2_found"] == 1) & ~totals["paper_id"].isin(done["paper_id"])].merge(papers[["paper_id", "posted"]], on="paper_id")
    todo = todo.sample(frac=1.0, random_state=42)                                # random order, reproducible
    rows = done.to_dict("records")
    for i, p in enumerate(tqdm(todo.itertuples(index=False), total=len(todo), desc="S2 12-month counts")):
        try:
            result = semantic_scholar.citations_in_first_year(p.s2_paper_id, date.fromisoformat(p.posted))
        except RuntimeError as e:                                               # one stubborn paper must not stop the run
            print(f"  skipped {p.paper_id}: {e}", file=sys.stderr, flush=True)
            continue
        rows.append({"paper_id": p.paper_id, **result})
        if (i + 1) % 200 == 0:                                                  # save progress every 200 papers
            pd.DataFrame(rows).to_parquet(out_path, index=False)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"exact 12-month counts from Semantic Scholar for {len(rows)} papers  ->  s2_cites12m.parquet", flush=True)


# --------------------------------------------------------------------------------------------- Hugging Face ----
def run_hf(papers: pd.DataFrame, prefix: str = "") -> None:
    ds = papers[(papers["field"] == "data_science") & (papers["source"] == "arxiv")]
    rows = [{"paper_id": p, **huggingface.paper(arxiv_id_of(p))} for p in tqdm(ds["paper_id"], desc="Hugging Face pages")]
    hf = pd.DataFrame(rows)
    hf.to_parquet(C.RAW_DIR / f"{prefix}hf_papers.parquet", index=False)
    print(f"Hugging Face has a page for {int(hf['hf_page'].sum())} of {len(hf)} data-science papers, "
          f"{int(hf['hf_repo'].notna().sum())} with a GitHub repository  ->  hf_papers.parquet", flush=True)


# ------------------------------------------------------------------------------------ bioRxiv / medRxiv ----
def run_rxiv(papers: pd.DataFrame, prefix: str = "") -> None:
    life = papers[papers["source"].isin(["biorxiv", "medrxiv"])]
    rows = []
    for p in tqdm(life.itertuples(index=False), total=len(life), desc="bioRxiv/medRxiv details"):
        try:
            d = biorxiv.details(p.source, p.paper_id)
        except RuntimeError as e:
            print(f"  skipped {p.paper_id}: {e}", file=sys.stderr, flush=True)
            d = None
        rows.append({"paper_id": p.paper_id, **(d or {"rxiv_category": None})})
    rx = pd.DataFrame(rows)
    rx.to_parquet(C.RAW_DIR / f"{prefix}rxiv_details.parquet", index=False)
    print(f"categories found for {int(rx['rxiv_category'].notna().sum())} of {len(rx)} life-science papers  ->  rxiv_details.parquet", flush=True)


def main(parts: list[str], prefix: str = "", table: str = "papers.csv") -> None:
    papers = load_papers(table)
    if "all" in parts:
        parts = ["s2", "hf", "rxiv"]
    for part in parts:
        {"s2": run_s2, "s2total": run_s2_total, "s2window": run_s2_window, "hf": run_hf, "rxiv": run_rxiv}[part](papers, prefix)


if __name__ == "__main__":
    args = sys.argv[1:]
    prefix = args[args.index("--prefix") + 1] if "--prefix" in args else ""
    table = args[args.index("--table") + 1] if "--table" in args else "papers.csv"
    parts = [a for a in args if not a.startswith("--") and a not in (prefix, table)]
    main(parts or ["all"], prefix, table)
