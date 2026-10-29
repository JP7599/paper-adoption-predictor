"""Iteration 3: the sealed hold-out set. Papers first posted January-June 2025, never looked at while developing.

    .venv/bin/python -m paper_adoption.holdout

The same pipeline as the pilot, run once more with different months, bigger samples and the file prefix "holdout_":
    data/raw/holdout_*.parquet         the harvest (acquire.py) and, later, the Semantic Scholar outcomes (enrich.py)
    data/processed/holdout.csv         the tidy table (prepare.py), same columns as papers.csv
The 12 months after a June 2025 posting end in June 2026, so every outcome in this set is complete today.
"""
from __future__ import annotations

from . import acquire, prepare
from . import config as C


def main() -> None:
    acquire.main(month_list=C.HOLDOUT_MONTHS, per_field=C.HOLDOUT_PER_FIELD, exact_per_field=0, prefix="holdout_")
    prepare.main(prefix="holdout_", out_name="holdout.csv")


if __name__ == "__main__":
    main()
