"""A sealed exam set for the biotech track: bioRxiv preprints of January-June 2025 in the five biotech categories.

    .venv/bin/python -m paper_adoption.exam_biotech                                            # harvest, outcomes, prepare, author track record, prepare again
    .venv/bin/python -m paper_adoption.embed specter2 exam_biotech_ exam_biotech.csv biotech   # SPECTER2 embeddings of the exam papers

Why: every model of the biotech track was compared and tuned on the 2024 papers, so 2024 can no longer be a clean test.
These 2025 papers were harvested after all decisions were made and are touched once, at the end of iteration five
(notebook 11). Same pipeline as the full harvest (harvest_full.py), prefix "exam_".
Files: data/raw/exam_papers_rxiv.parquet, exam_papers_harvest.parquet, exam_outcomes.parquet, exam_author_prestige.parquet
and data/processed/exam_biotech.csv.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from tqdm import tqdm

from . import authors, biorxiv, harvest_full, prepare
from . import config as C

PREFIX = "exam_"
MONTHS = [(2025, m) for m in range(1, 7)]           # January to June 2025
TABLE = "exam_biotech.csv"


def run_rxiv() -> None:
    rows = []
    for cat in tqdm(C.BIOTECH_CATEGORIES, desc="bioRxiv categories, 2025 H1"):
        for y, m in MONTHS:
            a, b = harvest_full.month_bounds(y, m)
            rows += biorxiv.first_versions(biorxiv.list_interval("biorxiv", a, b, category=cat), "biorxiv", "biotech")
    bio = pd.DataFrame(rows).drop_duplicates("paper_id")
    print(f"biotech exam set (bioRxiv, 5 categories, 2025 H1): {len(bio)} papers", flush=True)
    bio.to_parquet(C.RAW_DIR / f"{PREFIX}papers_rxiv.parquet", index=False)


def main() -> None:
    harvest_full.PREFIX = PREFIX                    # reuse the full-harvest steps with the exam prefix
    run_rxiv()
    harvest_full.run_outcomes()                     # OpenAlex records and first-year citations
    prepare.main(prefix=PREFIX, out_name=TABLE)     # the tidy table (without the author track record yet)
    authors.main(PREFIX, TABLE)                     # track record of first and last authors, years before posting
    prepare.main(prefix=PREFIX, out_name=TABLE)     # the tidy table again, now with the track record
    print("EXAM TABLE DONE", flush=True)


if __name__ == "__main__":
    main()
