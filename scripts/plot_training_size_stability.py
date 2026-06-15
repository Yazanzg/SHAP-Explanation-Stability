"""TASK 8: figures for attribution stability and performance vs training size.

Reads ONLY the frozen sorted result set produced by the learning-curve runner
and summariser:

  outputs/training_size_runs.csv
  outputs/training_size_stability_summary.csv
  outputs/training_size_performance_summary.csv
  outputs/importance_vectors/training_size_importance_vectors.csv

Main figures use the random-subset regime (config.plotting.main_figure_sizes,
N=80-280). The full-pool boundary size (config.plotting.boundary_size, N=320) is
excluded from main panels because participant-subset variation collapses there;
it is retained only in tables/appendix.

Figures written to outputs/figures/ as .png and .pdf, plus a self-contained
caption file figure_captions.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.config import FEATURE_COLUMNS, MAIN_FIGURE_SIZES, BOUNDARY_SIZE

MODEL_ORDER = ["logistic_regression", "random_forest", "svm_rbf"]
MODEL_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM-RBF",
}
MODEL_COLOR = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "svm_rbf": "#d62728",
}
MODEL_MARKER = {"logistic_regression": "o", "random_forest": "s", "svm_rbf": "^"}

LO, HI = config.PERCENTILE_INTERVAL  # e.g. 2.5 / 97.5 -> 95% band


def _save(fig: plt.Figure, stem: str) -> list[str]:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        out.append(str(path))
    plt.close(fig)
    return out


def _band_line_figure(
    summary: pd.DataFrame,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    """Line + vertical percentile-interval error bars, with a small per-model
    x-dodge so overlapping intervals stay readable."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    dodge = {"logistic_regression": -3.5, "random_forest": 0.0, "svm_rbf": 3.5}
    for model in MODEL_ORDER:
        d = summary[(summary.model == model) & (summary.training_size.isin(MAIN_FIGURE_SIZES))]
        d = d.sort_values("training_size")
        x = d.training_size.to_numpy(dtype=float) + dodge[model]
        mean = d[mean_col].to_numpy(dtype=float)
        yerr = np.vstack([mean - d[lo_col].to_numpy(dtype=float),
                          d[hi_col].to_numpy(dtype=float) - mean])
        ax.errorbar(x, mean, yerr=yerr, marker=MODEL_MARKER[model], color=MODEL_COLOR[model],
                    label=MODEL_LABEL[model], linewidth=1.8, markersize=6,
                    capsize=3, elinewidth=1.1, alpha=0.9)
    ax.set_xlabel("Training set size (balanced HC/CI)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(MAIN_FIGURE_SIZES)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    stab = pd.read_csv(config.OUTPUT_DIR / "training_size_stability_summary.csv")
    perf = pd.read_csv(config.OUTPUT_DIR / "training_size_performance_summary.csv")
    imp = pd.read_csv(config.IMPORTANCE_VECTORS_DIR / "training_size_importance_vectors.csv")

    created: list[str] = []

    # --- Figure 1: attribution stability (Pearson) ---
    created += _save(
        _band_line_figure(
            stab, "pearson_mean", "pearson_lo", "pearson_hi",
            ylabel="Attribution stability (pairwise Pearson of mean-|SHAP|)",
            title="SHAP attribution stability vs. training size",
        ),
        "fig1_attribution_stability_pearson",
    )

    # --- Figure 2: performance (AUROC) ---
    created += _save(
        _band_line_figure(
            perf, "auroc_mean", "auroc_lo", "auroc_hi",
            ylabel="Dev-set AUROC",
            title="Predictive performance (AUROC) vs. training size",
        ),
        "fig2_performance_auroc",
    )

    # --- Figure 3: attribution stability (Jaccard@5) ---
    created += _save(
        _band_line_figure(
            stab, "jaccard5_mean", "jaccard5_lo", "jaccard5_hi",
            ylabel="Top-5 feature overlap (Jaccard@5)",
            title="Top-5 SHAP feature-set stability vs. training size",
        ),
        "fig3_attribution_stability_jaccard5",
    )

    # --- Figure 4: per-model feature-importance trajectories (top features) ---
    imp_main = imp[imp.training_size.isin(MAIN_FIGURE_SIZES)]
    cell_mean = (
        imp_main.groupby(["model", "training_size", "feature_name"])["mean_abs_shap"]
        .mean()
        .reset_index()
    )
    top_k = 6
    for model in MODEL_ORDER:
        d = cell_mean[cell_mean.model == model]
        pivot = d.pivot(index="feature_name", columns="training_size", values="mean_abs_shap")
        pivot = pivot.reindex([f for f in FEATURE_COLUMNS if f in pivot.index])
        avg_imp = pivot.mean(axis=1).sort_values(ascending=False)
        top_feats = avg_imp.head(top_k).index.tolist()

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        cmap = plt.get_cmap("tab10")
        for i, feat in enumerate(top_feats):
            row = pivot.loc[feat].sort_index()
            ax.plot(row.index.to_numpy(), row.to_numpy(), marker="o", linewidth=1.6,
                    color=cmap(i % 10), label=feat)
        ax.set_xlabel("Training set size (balanced HC/CI)")
        ax.set_ylabel("Mean |SHAP| (averaged over 20 repeats)")
        ax.set_title(f"Feature-importance trajectories — {MODEL_LABEL[model]} (top {top_k})")
        ax.set_xticks(MAIN_FIGURE_SIZES)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        created += _save(fig, f"fig4_feature_importance_trajectory_{model}")

    # --- Captions ---
    band_txt = (f"Shaded bands / intervals are {int(HI - LO)}% percentile intervals "
                f"({LO:g}th-{HI:g}th percentile) computed across repeats.")
    boundary_txt = (f"Main figures use N=80-280; N={BOUNDARY_SIZE} is excluded from all main "
                    f"figures because it is the full-pool boundary: at the full training pool "
                    f"every repeat draws the same participants, so participant-subset variation "
                    f"collapses and the metric is degenerate (reported in tables/appendix only).")
    captions = {
        "fig1_attribution_stability_pearson":
            ("Figure 1. SHAP attribution stability versus training-set size for Logistic "
             "Regression, Random Forest, and SVM-RBF over N=80-280. Stability is the mean "
             "pairwise Pearson correlation between the per-repeat mean-|SHAP| feature-importance "
             f"vectors (13 features), over all C(20,2)=190 repeat pairs per cell. {band_txt} "
             "Higher values indicate more reproducible attributions across resampled training "
             f"sets. {boundary_txt}"),
        "fig2_performance_auroc":
            ("Figure 2. Predictive performance (dev-set AUROC) versus training-set size for "
             "Logistic Regression, Random Forest, and SVM-RBF over N=80-280. Lines show the mean "
             f"over 20 repeats per training size on the fixed 80-participant dev split. {band_txt} "
             f"{boundary_txt}"),
        "fig3_attribution_stability_jaccard5":
            ("Figure 3. Top-5 SHAP feature-set stability (Jaccard@5) versus training-set size for "
             "Logistic Regression, Random Forest, and SVM-RBF over N=80-280. Jaccard@5 is the mean "
             "overlap of the top-5 features (by mean-|SHAP|) between repeat pairs, over all "
             f"C(20,2)=190 pairs per cell. {band_txt} {boundary_txt}"),
    }
    for model in MODEL_ORDER:
        captions[f"fig4_feature_importance_trajectory_{model}"] = (
            f"Figure 4 ({MODEL_LABEL[model]}). Mean-|SHAP| trajectories of the top {top_k} "
            f"features (by average importance across N=80-280) for {MODEL_LABEL[model]}. Each "
            "point is the mean absolute SHAP value averaged over the 20 repeats at that training "
            "size, on the fixed dev split. Shows how the model's most-attributed features evolve "
            f"as the training set grows. {boundary_txt}")

    cap_path = config.FIGURES_DIR / "figure_captions.md"
    with cap_path.open("w", encoding="utf-8") as fh:
        fh.write("# TASK 8 figure captions\n\n")
        for stem, text in captions.items():
            fh.write(f"## {stem}\n\n{text}\n\n")
    created.append(str(cap_path))

    print(f"[plot] main sizes={MAIN_FIGURE_SIZES}  boundary (excluded)={BOUNDARY_SIZE}")
    print(f"[plot] wrote {len(created)} files to {config.FIGURES_DIR}:")
    for f in created:
        print("  ", Path(f).name)


if __name__ == "__main__":
    main()
