"""Every project decision as a named constant. Change things here, not inside the scripts."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # the project folder (.../Paper Adoption Predictor)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                          # answers from the APIs, tidied but not yet interpreted
CACHE_DIR = RAW_DIR / "_cache"                      # every single HTTP answer, so re-runs cost nothing
PROCESSED_DIR = DATA_DIR / "processed"              # the one tidy table the notebook reads
FIGURES_DIR = ROOT / "reports" / "figures"

load_dotenv(ROOT / ".env")                          # your e-mail and keys live in .env, never in code or output
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO") or None
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY") or None
S2_API_KEY = os.getenv("S2_API_KEY") or None                # Semantic Scholar: a steady 1 request/second instead of the shared pool
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or None

# ---- study window ---------------------------------------------------------------------------------------------
YEARS = [2022, 2023, 2024]                          # papers first posted in these years
OUTCOME_WINDOW_DAYS = 365                           # "adoption at 12 months": citations dated within a year of posting

# ---- iteration-zero pilot sizes (the proposal's full sizes are 10k-30k per field) --------------------------------
PILOT_PER_FIELD = 1000                              # papers we aim for per field
ARXIV_ANCHOR_DAYS = [1, 15]                         # arXiv cannot sample at random, so: two snapshots per month...
ARXIV_WINDOW_DAYS = 3                               # ...each taking the first papers submitted from the anchor day on
EXACT_CITES_PER_FIELD = 50                          # papers per field for which the exact 12-month count is fetched

# ---- the five fields ------------------------------------------------------------------------------------------
# arXiv fields: we query the categories (a * means "and all sub-categories") and keep a paper only when its PRIMARY
# category starts with one of the prefixes, so a physics paper cross-listed to cs.LG does not become data science.
# bioRxiv and medRxiv are read through OpenAlex (their own API was down when this was written); we sample among all
# works OpenAlex lists as hosted by the server.
FIELDS = {
    "data_science": {"source": "arxiv", "label": "arXiv cs.LG/cs.CL/cs.CV/stat.ML",
                     "query_categories": ["cs.LG", "cs.CL", "cs.CV", "stat.ML"],
                     "primary_prefixes": ["cs.LG", "cs.CL", "cs.CV", "stat.ML"]},
    "physics":      {"source": "arxiv", "label": "arXiv cond-mat/quant-ph/physics",
                     "query_categories": ["cond-mat*", "quant-ph", "physics*"],
                     "primary_prefixes": ["cond-mat", "quant-ph", "physics"]},
    "economics":    {"source": "arxiv", "label": "arXiv econ/q-fin",
                     "query_categories": ["econ*", "q-fin*"],
                     "primary_prefixes": ["econ", "q-fin"]},
    "biotech":      {"source": "openalex", "label": "bioRxiv", "openalex_source_id": "S4306402567"},
    "medicine":     {"source": "openalex", "label": "medRxiv", "openalex_source_id": "S3005729997"},
}

# ---- iteration 1: enrichment -------------------------------------------------------------------------------------
# bioRxiv subject categories that define "biotech" in the proposal (lower case, as the bioRxiv API spells them)
BIOTECH_CATEGORIES = ["bioengineering", "synthetic biology", "genomics", "bioinformatics", "molecular biology"]
# features that are knowable on the posting day from the preprint record itself (no OpenAlex metadata involved)
SAFE_FEATURES = ["n_authors", "title_words", "abstract_words", "has_code_link", "n_categories"]
# features from OpenAlex's record of the paper, shown in iteration zero to encode "later published" (never a model input)
LEAKY_FEATURES = ["n_references", "n_institutions", "n_countries"]

# ---- iteration 3: sealed hold-out set -----------------------------------------------------------------------------
HOLDOUT_MONTHS = [(2025, m) for m in range(1, 7)]      # papers first posted January-June 2025; their 12 months are complete
HOLDOUT_PER_FIELD = 600
