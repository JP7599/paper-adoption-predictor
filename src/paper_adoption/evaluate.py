"""Evaluation helpers shared by the iteration-2 and iteration-3 notebooks.

The protocol, fixed before any model was trained:
  * out-of-time scoring: every half-year from 2023 H1 to 2024 H2 is scored by a model trained ONLY on earlier papers;
  * AUROC and precision-at-top-10 %, per field and overall, always next to the base rate;
  * uncertainty from the bootstrap: resample the test papers 500 times and read off the 2.5th and 97.5th percentiles.

Reference: Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12,
2825-2830. (All models and metrics in this project use scikit-learn.)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HALF_YEARS = ["2022 H1", "2022 H2", "2023 H1", "2023 H2", "2024 H1", "2024 H2"]


def rolling_folds(half_years: list[str] = HALF_YEARS, first_test: int = 2) -> list[tuple[list[str], str]]:
    """[(train half-years, test half-year), ...]: test 2023 H1 after training on 2022; test 2023 H2 after training on
    2022 + 2023 H1; and so on. The model never sees a paper from the test half-year or later."""
    return [(half_years[:i], half_years[i]) for i in range(first_test, len(half_years))]


def out_of_time_scores(data: pd.DataFrame, fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
                       folds: list[tuple[list[str], str]] | None = None) -> pd.Series:
    """One score per paper from 2023 H1 on, each produced by a model that trained only on earlier half-years.
    `fit_predict(train_rows, test_rows)` must return one probability per test row. 2022 papers get no score (NaN):
    they are only ever used for training."""
    scores = pd.Series(np.nan, index=data.index, dtype=float)
    for train_halves, test_half in (folds or rolling_folds()):
        train = data[data["half_year"].isin(train_halves)]
        test = data[data["half_year"] == test_half]
        if len(train) and len(test):
            scores.loc[test.index] = fit_predict(train, test)
    return scores


def auroc_with_ci(y, score, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """AUROC and its 95 % bootstrap interval: resample the papers with replacement, recompute, repeat."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    point = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():                       # a resample needs both classes to define an AUROC
            values.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, lo, hi


def precision_at_top(y, score, share: float = 0.10) -> float:
    """Among the `share` of papers with the highest scores, the fraction that truly are positives."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    k = max(1, round(share * len(y)))
    top = np.argsort(-score)[:k]
    return float(y[top].mean())


def summarize(frame: pd.DataFrame, score_col: str, label: str, fields: list[str]) -> pd.DataFrame:
    """One row per field plus 'all fields': papers, base rate, AUROC with interval, precision at top 10 %.
    Only rows that have a score are used (2022 papers have none)."""
    scored = frame.dropna(subset=[score_col])
    rows = {}
    for name, g in [("all fields", scored)] + [(f, scored[scored["field"] == f]) for f in fields]:
        if len(g) == 0 or g[label].nunique() < 2:
            continue
        auc, lo, hi = auroc_with_ci(g[label], g[score_col])
        rows[name] = {"papers": len(g), "base rate": g[label].mean(), "AUROC": auc, "AUROC low": lo, "AUROC high": hi,
                      "precision at top 10 %": precision_at_top(g[label], g[score_col])}
    return pd.DataFrame(rows).T


def pretty(summary: pd.DataFrame) -> pd.DataFrame:
    """The same table with the interval written next to the AUROC, for reading."""
    out = summary.copy()
    out["AUROC (95 % interval)"] = [f"{a:.3f}  [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(out["AUROC"], out["AUROC low"], out["AUROC high"])]
    out["papers"] = out["papers"].astype(int)
    return out[["papers", "base rate", "AUROC (95 % interval)", "precision at top 10 %"]].round(3)
