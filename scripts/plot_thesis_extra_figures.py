"""Resubmission figure pack: corrected pipeline diagram, AUROC-with-chance-line,
performance-vs-stability dissociation, cohort composition, feature-correlation heatmap.

Writes PNG + PDF to outputs/figures/. Drop-in replacements:
  00_pipeline_overview.{png,pdf}   (removes Baselines/GridSearch; reframed)
  01_cohort_composition.{png,pdf}  (for Methods 3.1 next to Table 1)
  fig2_performance_auroc.{png,pdf} (adds 0.5 chance line)
New:
  fig5_performance_vs_stability.{png,pdf}
  fig6_feature_correlation_heatmap.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["logistic_regression", "random_forest", "svm_rbf"]
MODEL_LABEL = {"logistic_regression": "Logistic Regression",
               "random_forest": "Random Forest", "svm_rbf": "SVM-RBF"}
MODEL_COLOR = {"logistic_regression": "#1f77b4", "random_forest": "#2ca02c", "svm_rbf": "#d62728"}
MODEL_MARKER = {"logistic_regression": "o", "random_forest": "s", "svm_rbf": "^"}
MAIN_SIZES = [80, 120, 160, 200, 240, 280]

# Table-2 feature order (matches thesis)
FEATURES = ["word_count", "sentence_count", "type_token_ratio", "filler_count",
            "filler_ratio", "mean_clause_length", "content_density", "noun_ratio",
            "verb_ratio", "adjective_ratio", "adverb_ratio", "pronoun_ratio",
            "determiner_ratio"]


def save(fig, stem):
    paths = []
    for ext in ("png", "pdf"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p.name)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- 1. PIPELINE
def pipeline():
    BLUE, ORANGE, GRAY = "#cfe2f3", "#fce5cd", "#eeeeee"
    EDGE = {"fixed": "#3d6fb4", "varied": "#d98036", "step": "#888888"}
    boxes = [
        ("CTD\ntranscripts", "step"),
        ("Preprocess", "step"),
        ("13 linguistic\nfeatures", "step"),
        ("Predefined split\n320 train pool / 80 dev", "fixed"),
        ("Balanced subset\nof size $N$", "varied"),
        ("Sort by\nparticipant ID", "varied"),
        ("Fit fixed-\nconfiguration model", "varied"),
        ("SHAP on\nfixed dev split", "fixed"),
        ("mean-|SHAP|\nvector (13-dim)", "step"),
        ("190 pairwise\nPearson / Jaccard", "step"),
        ("Percentile bands,\nCSV, figures", "step"),
    ]
    fill = {"fixed": BLUE, "varied": ORANGE, "step": GRAY}
    # serpentine coords: row1 L->R (0..3), row2 R->L (3..0), row3 L->R (0..2)
    coords = [(0, 2), (1, 2), (2, 2), (3, 2),
              (3, 1), (2, 1), (1, 1), (0, 1),
              (0, 0), (1, 0), (2, 0)]
    bw, bh = 0.86, 0.62
    fig, ax = plt.subplots(figsize=(11, 4.6))
    centers = []
    for (label, kind), (cx, cy) in zip(boxes, coords):
        x, y = cx * 1.0, cy * 1.0
        centers.append((x, y))
        box = FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.6, edgecolor=EDGE[kind], facecolor=fill[kind], zorder=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=3)

    def arrow(a, b, rad=0.0):
        (x0, y0), (x1, y1) = centers[a], centers[b]
        cs = f"arc3,rad={rad}"
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=cs,
                     arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#555555",
                     shrinkA=26, shrinkB=26, zorder=1))

    # within-row straight arrows
    for a, b in [(0, 1), (1, 2), (2, 3),         # row1 ->
                 (4, 5), (5, 6), (6, 7),         # row2 <-
                 (8, 9), (9, 10)]:               # row3 ->
        arrow(a, b)
    # row turns (curved elbows)
    arrow(3, 4, rad=-0.45)   # box4 (3,2) down to box5 (3,1)
    arrow(7, 8, rad=-0.45)   # box8 (0,1) down to box9 (0,0)

    # dashed enclosure around the three "varies across repeats" boxes (row 2: x=1,2,3)
    rx0, rx1 = 1 - bw / 2 - 0.09, 3 + bw / 2 + 0.09
    ry0, ry1 = 1 - bh / 2 - 0.09, 1 + bh / 2 + 0.09
    ax.add_patch(FancyBboxPatch((rx0, ry0), rx1 - rx0, ry1 - ry0,
                 boxstyle="round,pad=0,rounding_size=0.1", fill=False,
                 linestyle=(0, (5, 4)), linewidth=1.4, edgecolor="#d98036", zorder=4))
    ax.text(2.0, ry1 + 0.07,
            "$\\times$20 repeats per (model $\\times$ size): only the sampled training subset varies",
            ha="center", va="bottom", fontsize=8, color="#b35f12")

    legend = [Patch(facecolor=BLUE, edgecolor=EDGE["fixed"], label="Held fixed across all runs"),
              Patch(facecolor=ORANGE, edgecolor=EDGE["varied"], label="Varies across the 20 repeats"),
              Patch(facecolor=GRAY, edgecolor=EDGE["step"], label="Shared pipeline step")]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5,
              bbox_to_anchor=(1.0, -0.04))

    ax.set_xlim(-0.7, 3.7)
    ax.set_ylim(-0.6, 2.7)
    ax.axis("off")
    fig.tight_layout()
    return save(fig, "00_pipeline_overview")


# ------------------------------------------------------------------ 2. AUROC + chance
def auroc_with_chance(perf):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    dodge = {"logistic_regression": -3.5, "random_forest": 0.0, "svm_rbf": 3.5}
    for m in MODEL_ORDER:
        d = perf[(perf.model == m) & (perf.training_size.isin(MAIN_SIZES))].sort_values("training_size")
        x = d.training_size.to_numpy(float) + dodge[m]
        mean = d.auroc_mean.to_numpy(float)
        yerr = np.vstack([mean - d.auroc_lo.to_numpy(float), d.auroc_hi.to_numpy(float) - mean])
        ax.errorbar(x, mean, yerr=yerr, marker=MODEL_MARKER[m], color=MODEL_COLOR[m],
                    label=MODEL_LABEL[m], linewidth=1.8, markersize=6, capsize=3,
                    elinewidth=1.1, alpha=0.9)
    ax.axhline(0.5, ls="--", lw=1.3, color="#666666", zorder=0)
    ax.text(MAIN_SIZES[-1], 0.508, "Chance (0.5)", ha="right", va="bottom",
            fontsize=8.5, color="#666666")
    ax.set_xlabel("Training set size (balanced HC/CI)")
    ax.set_ylabel("Dev-set AUROC")
    ax.set_title("Predictive performance (AUROC) vs. training size")
    ax.set_xticks(MAIN_SIZES)
    ax.set_ylim(0.28, 0.82)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, "fig2_performance_auroc")


# ----------------------------------------------- 3. performance vs stability (dissociation)
def perf_vs_stability(perf, stab):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, m in zip(axes, MODEL_ORDER):
        p = perf[(perf.model == m) & (perf.training_size.isin(MAIN_SIZES))].sort_values("training_size")
        s = stab[(stab.model == m) & (stab.training_size.isin(MAIN_SIZES))].sort_values("training_size")
        x = p.training_size.to_numpy(float)
        ax.plot(x, p.auroc_mean.to_numpy(float), marker="o", lw=1.9, color=MODEL_COLOR[m],
                label="AUROC (performance)")
        ax.plot(x, s.pearson_mean.to_numpy(float), marker="^", ls="--", lw=1.9,
                color=MODEL_COLOR[m], alpha=0.65, label="Pearson (attribution stability)")
        ax.set_title(MODEL_LABEL[m], fontsize=10)
        ax.set_xticks(MAIN_SIZES)
        ax.set_xlabel("Training set size")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score (0-1)")
    # one shared legend (line-style coded, model-neutral)
    h = [plt.Line2D([], [], color="#444", marker="o", lw=1.9, label="AUROC (performance)"),
         plt.Line2D([], [], color="#444", marker="^", ls="--", lw=1.9, alpha=0.7,
                    label="Pearson (attribution stability)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Performance does not track attribution stability", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return save(fig, "fig5_performance_vs_stability")


# ------------------------------------------------------------------ 4. cohort
def cohort(master):
    diag = master.diagnosis.value_counts()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    HC_C, CI_C = "#4477aa", "#cc6677"
    # panel 1: 3-class
    order3 = ["HC", "MCI", "Dementia"]
    vals3 = [int(diag.get(k, 0)) for k in order3]
    cols3 = [HC_C, CI_C, CI_C]
    b = axes[0].bar(order3, vals3, color=cols3, edgecolor="black", linewidth=0.6)
    axes[0].bar_label(b, fontsize=9)
    axes[0].set_title("Diagnostic labels (3-class)", fontsize=10)
    axes[0].set_ylabel("Participants")
    axes[0].set_ylim(0, 230)
    # panel 2: binary
    b2 = axes[1].bar(["HC", "CI"], [200, 200], color=[HC_C, CI_C], edgecolor="black", linewidth=0.6)
    axes[1].bar_label(b2, fontsize=9)
    axes[1].set_title("Binary target (MCI+dementia $\\rightarrow$ CI)", fontsize=10)
    axes[1].set_ylim(0, 230)
    # panel 3: split x class stacked
    splits = ["Train pool", "Dev split"]
    hc = [160, 40]; ci = [160, 40]
    axes[2].bar(splits, hc, color=HC_C, edgecolor="black", linewidth=0.6, label="HC")
    axes[2].bar(splits, ci, bottom=hc, color=CI_C, edgecolor="black", linewidth=0.6, label="CI")
    for i, (h_, c_) in enumerate(zip(hc, ci)):
        axes[2].text(i, h_ / 2, str(h_), ha="center", va="center", fontsize=9, color="white")
        axes[2].text(i, h_ + c_ / 2, str(c_), ha="center", va="center", fontsize=9, color="white")
        axes[2].text(i, h_ + c_ + 6, str(h_ + c_), ha="center", va="bottom", fontsize=9)
    axes[2].set_title("Predefined split $\\times$ class", fontsize=10)
    axes[2].set_ylim(0, 360)
    axes[2].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "01_cohort_composition")


# ------------------------------------------------------------- 5. feature correlation heatmap
def corr_heatmap(fc):
    M = fc.pivot(index="feature_a", columns="feature_b", values="pearson")
    M = M.reindex(index=FEATURES, columns=FEATURES)
    A = M.to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(FEATURES, fontsize=7.5)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = A[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "#222222")
    # highlight the type_token_ratio <-> word_count cell(s)
    ti, wi = FEATURES.index("type_token_ratio"), FEATURES.index("word_count")
    for (r, c) in [(ti, wi), (wi, ti)]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                     edgecolor="black", lw=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson correlation", fontsize=9)
    ax.set_title("Pairwise correlation of the 13 transcript-derived features", fontsize=11)
    fig.tight_layout()
    return save(fig, "fig6_feature_correlation_heatmap")


def main():
    perf = pd.read_csv(OUT / "training_size_performance_summary.csv")
    stab = pd.read_csv(OUT / "training_size_stability_summary.csv")
    master = pd.read_csv(OUT / "process2_ctd_binary_master.csv")
    fc = pd.read_csv(OUT / "audit" / "feature_correlations.csv")

    created = []
    created += pipeline()
    created += auroc_with_chance(perf)
    created += perf_vs_stability(perf, stab)
    created += cohort(master)
    created += corr_heatmap(fc)

    # sanity print of the headline correlation
    ttr_wc = fc[(fc.feature_a == "type_token_ratio") & (fc.feature_b == "word_count")].pearson
    print("type_token_ratio <-> word_count Pearson =", round(float(ttr_wc.iloc[0]), 3))
    print("wrote:", ", ".join(created))


if __name__ == "__main__":
    main()
