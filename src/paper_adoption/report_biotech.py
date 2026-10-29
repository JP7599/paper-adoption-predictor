"""Writes reports/biotech_track_results.md: the final scores of every biotech iteration, read from the CSV files the notebooks save.

    .venv/bin/python -m paper_adoption.report_biotech
"""
from __future__ import annotations

import pandas as pd

from . import config as C

REPORTS = C.ROOT / "reports"
ITERATIONS = [  # (number, notebook, what changed, csv, how to pick the final row)
    (0, "06_biotech_iteration_zero_knn_simple", "the simplest idea: k-nearest neighbours on 14 day-one facts", "biotech_iteration0_results.csv", None),
    (1, "07_biotech_iteration_one_simple", "straight-line models: linear and logistic regression on the same facts", "biotech_iteration1_results.csv", None),
    (2, "08_biotech_iteration_two_simple", "flexible algorithms: random forest and gradient boosting (same facts)", "biotech_iteration2_results.csv", "gradient"),
    (3, "09_biotech_iteration_three_text_simple", "new information: the words of title and abstract (TF-IDF) + the facts", "biotech_iteration3_results.csv", "words + facts"),
    (4, "10_biotech_iteration_four_specter2_simple", "a pretrained science model (SPECTER2) + facts + words", "biotech_iteration4_results.csv", "embeddings + facts + words"),
    (5, "11_biotech_iteration_five_exam_simple", "same model as 4: error bars, calibrated probabilities, sealed 2025 exam", "biotech_iteration5_results.csv", None),
    (7, "13_biotech_iteration_seven_unknown_labs_simple", "a model blind to the authors, for papers from unknown labs (sealed exam)", "biotech_iteration7_results.csv", "content-only"),
]


def final_row(csv: str, pick: str | None) -> dict:
    t = pd.read_csv(REPORTS / csv)
    if pick:
        t = t[t["algorithm"].str.contains(pick, regex=False)]
    r2col = "R2" if "R2" in t else "R2 (how many citations)" if "R2 (how many citations)" in t else None
    aucol = "AUROC" if "AUROC" in t else "AUROC (top 10 %, yes/no)"
    row = {"R2": t[r2col].dropna().iloc[0] if r2col else None, "AUROC": t[aucol].dropna().iloc[0], "precision": t["precision at top 10 %"].dropna().iloc[0]}
    if "recall of unknown-lab top papers" in t:
        row["note"] = f"catches {t['recall of unknown-lab top papers'].iloc[0]:.0%} of the top papers from unknown labs (overall model: 16 %)"
    if "AUROC low" in t:
        row["interval"] = f"[{t['AUROC low'].iloc[0]:.3f}, {t['AUROC high'].iloc[0]:.3f}]"
    return row


def main() -> None:
    lines = ["# Biotech track: results per iteration", "",
             "Every row is one notebook and one complete iteration: question, data, model, score, conclusion. "
             "Iterations 0-4 are scored on the 2024 papers (models learn from 2022-2023); iteration 5 scores the iteration-4 model once on "
             "bioRxiv papers of January-June 2025 that no decision was based on. All papers are bioRxiv preprints in five biotech categories "
             "(bioinformatics, molecular biology, genomics, bioengineering, synthetic biology); the label is the top 10 % of first-year "
             "citations within the paper's half-year.", "",
             "Compare with: R² 0 = guessing the average; AUROC 0.5 = coin flip; precision at top 10 %: 0.10 = random reading.", "",
             "| iteration | what changed | R² | AUROC | precision at top 10 % | notebook |", "|---|---|---|---|---|---|"]
    for n, nb, what, csv, pick in ITERATIONS:
        if not (REPORTS / csv).exists():
            continue
        r = final_row(csv, pick)
        auroc = f"{r['AUROC']:.3f}" + (f" {r['interval']}" if "interval" in r else "")
        r2 = f"{r['R2']:.3f}" if r["R2"] is not None else "–"
        what_full = what + (f" · {r['note']}" if "note" in r else "")
        lines.append(f"| {n} | {what_full} | {r2} | {auroc} | {r['precision']:.3f} | `notebooks/{nb}.ipynb` |")
    for title, csv, cols in [("What adds value (notebook 14, ablation on the sealed exam)", "biotech_ablation_results.csv", ["model", "AUROC", "low", "high", "precision at top 10 %", "of the 50 most cited, found"]),
                             ("Why logistic regression (notebook 14, same inputs, five algorithms)", "biotech_bakeoff_results.csv", ["algorithm", "AUROC on the exam", "low", "high", "AUROC on 2024", "seconds to train", "gives probabilities", "explainable weights"])]:
        f = REPORTS / csv
        if f.exists():
            t = pd.read_csv(f)[cols]
            lines += ["", f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    lines += ["", "Reading the ablation: each ingredient alone is about 0.68; a text reading plus the facts is about 0.72; SPECTER2 and TF-IDF are interchangeable "
              "ways to read the text, so the model is not \"just SPECTER2\". Reading the bake-off: logistic regression ties the best and is the only one that gives probabilities, "
              "prints its weights and takes the 30,000 word columns."]
    lines += ["", "The check a reader asks for (notebook 11): of the 50 most cited papers of January-June 2025, 32 were on the tool's top-10 % list made from "
              "day-one information (a random list of that size holds about 5); the top-20 % list held 43 and the top-30 % list 46.", "",
              "Iteration 6 turned the iteration-4 model into a tool (`python -m paper_adoption.predict`, web demo `app/biotech_app.py`): "
              "a new preprint in, a calibrated probability with reasons and the five most similar past papers out, plus the rising topics; "
              "notebook 12 scores preprints posted this month and files the predictions to be checked in a year.", "",
              "![the whole track](figures/bio5_whole_track.png)", "",
              "The story in one sentence: iterations 0-2 changed the algorithm and the scores barely moved; iteration 3 changed the information "
              "(the text) and the scores jumped; iteration 4 added a pretrained science model for a small extra; iteration 5 showed what holds "
              "up on papers nobody had looked at, with error bars and probabilities that mean what they say.", "",
              "Charts follow IBCS (one message per chart, grey bars, labels at the bar ends, dotted reference lines). "
              "Generated by `python -m paper_adoption.report_biotech` from the CSV files in this folder."]
    (REPORTS / "biotech_track_results.md").write_text("\n".join(lines) + "\n")
    print("wrote", REPORTS / "biotech_track_results.md")


if __name__ == "__main__":
    main()
