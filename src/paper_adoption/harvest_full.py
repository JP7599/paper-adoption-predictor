"""The full harvest (dataset v1): the proposal's sample sizes, built with the same pipeline as the pilot.

    .venv/bin/python -m paper_adoption.harvest_full arxiv       # 3 arXiv fields, one request per day (hours; resumable)
    .venv/bin/python -m paper_adoption.harvest_full rxiv        # bioRxiv by the 5 biotech categories + medRxiv (about an hour)
    .venv/bin/python -m paper_adoption.harvest_full outcomes    # OpenAlex records for every paper, keyed batch look-ups (minutes)
    .venv/bin/python -m paper_adoption.harvest_full prepare     # data/processed/full_papers.csv
    then:  enrich s2total / s2window / hf  --prefix full_ --table full_papers.csv ;  authors full_ full_papers.csv

Files: data/raw/full_papers_harvest.parquet (all harvested papers), full_outcomes.parquet, and the enrichment files
with the prefix full_. Everything is cached, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd
from tqdm import tqdm

from . import acquire, biorxiv, prepare
from . import config as C

PREFIX = "full_"
START, END = date(2022, 1, 1), date(2024, 12, 31)
TARGET = {"data_science": 30000, "physics": 25000, "economics": 10000}      # arXiv fields: papers per day = target / 1096 days
MEDRXIV_PAGES_PER_WINDOW = 2                                                   # 2-day windows, 2 pages of 30 -> ~50 new papers


def months():
    return [(y, m) for y in C.YEARS for m in range(1, 13)]


def month_bounds(y, m):
    first = date(y, m, 1)
    return first, acquire.last_day_of(first)


def run_arxiv() -> None:
    n_days = (END - START).days + 1
    frames = []
    for field, spec in C.FIELDS.items():
        if spec["source"] != "arxiv":
            continue
        per_day = -(-TARGET[field] // n_days)                                   # ceiling division: 30000 / 1096 -> 28
        df = acquire.harvest_arxiv_daily(field, spec, START, END, per_day, request_n=max(2 * per_day, 40))
        df.to_parquet(C.RAW_DIR / f"{PREFIX}papers_arxiv_{field}.parquet", index=False)
        print(f"{field}: {len(df)} papers", flush=True)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(C.RAW_DIR / f"{PREFIX}papers_arxiv.parquet", index=False)


def run_rxiv() -> None:
    rows = []
    for cat in tqdm(C.BIOTECH_CATEGORIES, desc="bioRxiv categories"):
        for y, m in months():
            a, b = month_bounds(y, m)
            rows += biorxiv.first_versions(biorxiv.list_interval("biorxiv", a, b, category=cat), "biorxiv", "biotech")
    bio = pd.DataFrame(rows).drop_duplicates("paper_id")
    print(f"biotech (bioRxiv, 5 categories): {len(bio)} papers", flush=True)
    rows = []
    day = START
    windows = []
    while day <= END:
        windows.append((day, min(END, day.replace() + pd.Timedelta(days=1).to_pytimedelta())))
        day = day + pd.Timedelta(days=2).to_pytimedelta()
    for a, b in tqdm(windows, desc="medRxiv 2-day windows"):
        rows += biorxiv.first_versions(biorxiv.list_interval("medrxiv", a, b, max_pages=MEDRXIV_PAGES_PER_WINDOW), "medrxiv", "medicine")
    med = pd.DataFrame(rows).drop_duplicates("paper_id")
    print(f"medicine (medRxiv, all categories): {len(med)} papers", flush=True)
    pd.concat([bio, med], ignore_index=True).to_parquet(C.RAW_DIR / f"{PREFIX}papers_rxiv.parquet", index=False)


def run_outcomes() -> None:
    parts = [pd.read_parquet(C.RAW_DIR / f"{PREFIX}{name}.parquet") for name in ("papers_arxiv", "papers_rxiv")
             if (C.RAW_DIR / f"{PREFIX}{name}.parquet").exists()]
    papers = pd.concat(parts, ignore_index=True).drop_duplicates("paper_id")
    papers.to_parquet(C.RAW_DIR / f"{PREFIX}papers_harvest.parquet", index=False)
    print(f"harvested papers in total: {len(papers)}", flush=True)
    out = []
    for i in tqdm(range(0, len(papers), 2000), desc="OpenAlex outcomes (batches of 2,000 papers)"):
        out.append(acquire.outcomes_for_papers(papers.iloc[i:i + 2000]))
    outcomes = pd.concat(out, ignore_index=True)
    outcomes.to_parquet(C.RAW_DIR / f"{PREFIX}outcomes.parquet", index=False)
    print(f"found in OpenAlex: {int(outcomes['matched'].sum())} of {len(outcomes)}", flush=True)


def run_prepare() -> None:
    prepare.main(prefix=PREFIX, out_name="full_papers.csv")


def main(parts: list[str]) -> None:
    steps = {"arxiv": run_arxiv, "rxiv": run_rxiv, "outcomes": run_outcomes, "prepare": run_prepare}
    for part in (["arxiv", "rxiv", "outcomes", "prepare"] if "all" in parts else parts):
        steps[part]()


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
