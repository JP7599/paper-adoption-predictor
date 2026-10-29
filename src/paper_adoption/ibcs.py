"""Chart rules for the notebooks, following IBCS (International Business Communication Standards).

The rules we apply (the IBCS "SUCCESS" rules: Say, Unify, Condense, Check, Express, Simplify, Structure):
* one message per chart, written as the title; a grey subtitle says measure, unit and period
* dark grey for the item we talk about, light grey for the others; colour only when it carries meaning
  (green = better, red = worse in variance charts; one colour per category on a map)
* a value label at the end of every bar, so no value axis, no gridlines and no legend are needed
* bars start at zero; a dotted reference line marks the honest comparison (coin flip, random reading, zero)
* horizontal bars compare items (algorithms, inputs, words); vertical columns show a sequence (iterations, k)
* the same measure always uses the same scale, so charts can be compared across notebooks

Reference: IBCS Association. (2021). International Business Communication Standards, version 1.2. https://www.ibcs.com/standards
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

DARK, MID, LIGHT, TEXT = "#404040", "#8C8C8C", "#C8C8C8", "#262626"
GOOD, BAD = "#2E7D32", "#C62828"
SCALES = {"AUROC": (0.0, 1.0), "R2": (0.0, 0.20), "precision": (0.0, 0.50)}   # the same measure, the same scale, everywhere
CATEGORY_COLOURS = ["#1F77B4", "#D97A00", "#2CA02C", "#C62828", "#7B4FA6", "#8C564B"]


def style() -> None:
    """Apply the rules to every chart made after this call."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.edgecolor": MID, "axes.linewidth": 0.8, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 9,
    })


def title(ax, message: str, subtitle: str | None = None) -> None:
    """The message in bold, the measure/unit/period in grey below it."""
    ax.set_title("")
    ax.annotate(message, xy=(0, 1), xycoords="axes fraction", xytext=(0, 18 if subtitle else 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=11, fontweight="bold", color=TEXT, annotation_clip=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9, color=MID, annotation_clip=False)


def _clean(ax, horizontal: bool) -> None:
    if horizontal:
        ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.yaxis.set_visible(False); ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def bars(ax, labels, values, highlight=None, baseline: float | None = None, baseline_label: str | None = None,
         fmt: str = "{:.3f}", scale: tuple[float, float] | None = None, horizontal: bool = True, colors=None, err=None) -> None:
    """Bars from zero with the value at the end. `highlight` (a label or a list) is dark, the rest light grey.
    `err` = optional list of (low, high) bounds per bar, drawn as thin whiskers (for example a 95 % interval)."""
    labels, values = list(labels), np.asarray(values, dtype=float)
    err = None if err is None else np.asarray(err, dtype=float)
    if err is not None:
        err = np.where(np.isnan(err), values[:, None], err)   # a NaN row = no whisker for that bar
    ends = values if err is None else np.where(values >= 0, np.maximum(values, err[:, 1]), np.minimum(values, err[:, 0]))
    if colors is None:
        if highlight is None:
            colors = [DARK] * len(labels)
        else:
            hl = set(highlight) if isinstance(highlight, (list, tuple, set)) else {highlight}
            colors = [DARK if lab in hl else LIGHT for lab in labels]
    lo = min(0.0, float(ends.min()) * 1.15)
    hi = max(0.0, float(ends.max()) * 1.15)
    if scale is not None:
        lo, hi = min(lo, scale[0]), max(hi, scale[1])
    if ends.min() < 0:                                          # room for the label left of a negative bar, clear of the category names
        lo = ends.min() - 0.22 * (hi - ends.min())
    pad = (hi - lo) * 0.012
    if horizontal:
        pos = np.arange(len(labels))[::-1]                     # first item on top
        ax.barh(pos, values, color=colors, height=0.62)
        if err is not None:
            m = err[:, 1] > err[:, 0]                                                 # whiskers only where there is a real interval
            ax.errorbar(values[m], pos[m], xerr=[values[m] - err[m, 0], err[m, 1] - values[m]], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
        for p, v, e in zip(pos, values, ends):
            ax.text(e + (pad if v >= 0 else -pad), p, fmt.format(v), va="center", ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
        ax.set_yticks(pos); ax.set_yticklabels(labels)
        ax.set_xlim(lo, hi)
        if baseline is not None:
            ax.axvline(baseline, color=MID, linewidth=1, linestyle=(0, (3, 3)))
            if baseline_label:
                ax.set_ylim(-1.0, len(labels) - 0.4)
                ax.text(baseline, -0.6, " " + baseline_label, fontsize=8.5, color=MID, va="top", ha="left")
        ax.axvline(0, color=MID, linewidth=0.8)
    else:
        pos = np.arange(len(labels))
        ax.bar(pos, values, color=colors, width=0.62)
        if err is not None:
            m = err[:, 1] > err[:, 0]
            ax.errorbar(pos[m], values[m], yerr=[values[m] - err[m, 0], err[m, 1] - values[m]], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
        for p, v, e in zip(pos, values, ends):
            ax.text(p, e + (pad if v >= 0 else -pad), fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=TEXT)
        ax.set_xticks(pos); ax.set_xticklabels(labels)
        ax.set_ylim(lo, hi)
        if baseline is not None:
            ax.axhline(baseline, color=MID, linewidth=1, linestyle=(0, (3, 3)))
            if baseline_label:
                ax.text(-0.45, baseline, baseline_label, fontsize=8.5, color=MID, va="bottom", ha="left")
        ax.axhline(0, color=MID, linewidth=0.8)
    _clean(ax, horizontal)


def variance(ax, labels, deltas, fmt: str = "{:+.3f}", horizontal: bool = False) -> None:
    """Change versus the previous item: green when better, red when worse, from zero."""
    colors = [GOOD if d >= 0 else BAD for d in deltas]
    bars(ax, labels, deltas, fmt=fmt, horizontal=horizontal, colors=colors)


def line(ax, x, y, mark=None, fmt: str = "{:.3f}", baseline: float | None = None, baseline_label: str | None = None,
         logx: bool = False, scale: tuple[float, float] | None = None) -> None:
    """One dark line with points; the chosen point (`mark`, an x value) gets its value written next to it."""
    x, y = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    ax.plot(x, y, color=DARK, linewidth=1.8, marker="o", markersize=4)
    if mark is not None:
        i = int(np.argmin(np.abs(x - mark)))
        ax.plot([x[i]], [y[i]], marker="o", markersize=8, color=DARK)
        ax.annotate(fmt.format(y[i]) + f"  (k = {int(x[i])})", xy=(x[i], y[i]), xytext=(6, 8), textcoords="offset points", fontsize=9, color=TEXT)
    if baseline is not None:
        ax.axhline(baseline, color=MID, linewidth=1, linestyle=(0, (3, 3)))
        if baseline_label:
            ax.text(x.max(), baseline, baseline_label + " ", fontsize=8.5, color=MID, va="bottom", ha="right")
    if logx:
        ax.set_xscale("log")
    if scale is not None:
        ax.set_ylim(*scale)
    ax.tick_params(length=0)


def scatter(ax, x, y, diagonal: bool = True, xlabel: str = "", ylabel: str = "") -> None:
    """Points in grey with the perfect-prediction diagonal."""
    ax.scatter(x, y, s=6, alpha=0.3, color=DARK, linewidths=0)
    if diagonal:
        lim = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        ax.plot(lim, lim, color=MID, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.tick_params(length=0)


def roc(ax, curves, highlight: str | None = None) -> None:
    """ROC curves: `curves` is a list of (name, fpr, tpr, auroc); the highlighted one is dark, the rest grey."""
    greys = [MID, LIGHT, "#A8A8A8", "#6E6E6E"]
    g = 0
    for name, fpr, tpr, auroc in curves:
        if highlight is None or name == highlight:
            ax.plot(fpr, tpr, color=DARK, linewidth=2.2, label=f"{name}  {auroc:.3f}")
        else:
            ax.plot(fpr, tpr, color=greys[g % len(greys)], linewidth=1.6, label=f"{name}  {auroc:.3f}"); g += 1
    ax.plot([0, 1], [0, 1], color=MID, linewidth=1, linestyle=(0, (3, 3)), label="coin flip  0.500")
    ax.set_xlabel("false alarm rate"); ax.set_ylabel("share of real top papers found")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(length=0)
    ax.legend(loc="lower right", title="AUROC", title_fontsize=9, alignment="left")


def hist(ax, values, bins, xlabel: str = "", ylabel: str = "number of papers") -> None:
    ax.hist(values, bins=bins, color=DARK)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.tick_params(length=0)


def finish(fig, path=None) -> None:
    """Tight layout, save as png if a path is given, show."""
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
