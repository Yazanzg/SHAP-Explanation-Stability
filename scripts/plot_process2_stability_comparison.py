"""
Publication-quality figure: direct SHAP vs bootstrap explanation stability.

Outputs (FIGURES_DIR from src.config):
  process2_stability_direct_vs_bootstrap.png  (300 dpi)
  process2_stability_direct_vs_bootstrap.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR

# ---------------------------------------------------------------------------
# Exact values (do not modify)
# ---------------------------------------------------------------------------
MODELS = ["Logistic Regression", "Random Forest", "SVM RBF"]
MODELS_SHORT = ["LR", "RF", "SVM"]

# Direct SHAP stability (baseline vs optimized comparison on fixed dev SHAP)
DIRECT_SPEARMAN = [0.978, 0.945, 0.846]
DIRECT_JACCARD5 = [1.000, 1.000, 0.667]

# Bootstrap stability (pairwise mean across bootstrap iterations)
BOOT_SPEARMAN_BASELINE = [0.429, 0.512, 0.432]
BOOT_SPEARMAN_OPTIMIZED = [0.366, 0.562, 0.393]
BOOT_JACCARD5_BASELINE = [0.444, 0.463, 0.396]
BOOT_JACCARD5_OPTIMIZED = [0.387, 0.464, 0.432]

Y_MIN, Y_MAX = 0.0, 1.05


def _add_bar_labels(ax, bars, values: list[float], fmt: str = "{:.3f}") -> None:
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#cccccc",
            "grid.linewidth": 0.5,
        }
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6.5, 7.0),
        sharex=False,
        gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.32},
    )

    n_models = len(MODELS)
    x = np.arange(n_models)
    width_direct = 0.32

    # --- Top panel: Direct SHAP stability ---
    bars_sp = ax_top.bar(
        x - width_direct / 2,
        DIRECT_SPEARMAN,
        width_direct,
        label="Spearman",
        color="#4C72B0",
        edgecolor="#333333",
        linewidth=0.5,
    )
    bars_j5 = ax_top.bar(
        x + width_direct / 2,
        DIRECT_JACCARD5,
        width_direct,
        label="Jaccard@5",
        color="#55A868",
        edgecolor="#333333",
        linewidth=0.5,
    )
    _add_bar_labels(ax_top, bars_sp, DIRECT_SPEARMAN)
    _add_bar_labels(ax_top, bars_j5, DIRECT_JACCARD5)

    ax_top.set_ylabel("Stability score")
    ax_top.set_title("Direct SHAP Stability", fontweight="bold", pad=8)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(MODELS, rotation=0, ha="center")
    ax_top.set_ylim(Y_MIN, Y_MAX)
    ax_top.set_yticks(np.arange(0, 1.1, 0.2))
    ax_top.yaxis.grid(True, linestyle="-", alpha=0.4)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="upper right", frameon=True, framealpha=1, edgecolor="#cccccc")

    # --- Bottom panel: Bootstrap stability (4 bars per model) ---
    width_boot = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width_boot

    series = [
        (BOOT_SPEARMAN_BASELINE, "Baseline Spearman", "#4C72B0"),
        (BOOT_SPEARMAN_OPTIMIZED, "Optimized Spearman", "#8172B2"),
        (BOOT_JACCARD5_BASELINE, "Baseline Jaccard@5", "#55A868"),
        (BOOT_JACCARD5_OPTIMIZED, "Optimized Jaccard@5", "#C44E52"),
    ]

    for offset, (values, label, color) in zip(offsets, series):
        bars = ax_bot.bar(
            x + offset,
            values,
            width_boot,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
        )
        _add_bar_labels(ax_bot, bars, values)

    ax_bot.set_ylabel("Stability score")
    ax_bot.set_title("Bootstrap-Based Stability", fontweight="bold", pad=8)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(MODELS, rotation=0, ha="center")
    ax_bot.set_ylim(Y_MIN, Y_MAX)
    ax_bot.set_yticks(np.arange(0, 1.1, 0.2))
    ax_bot.yaxis.grid(True, linestyle="-", alpha=0.4)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=1,
        edgecolor="#cccccc",
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.08, hspace=0.38)

    png_path = FIGURES_DIR / "process2_stability_direct_vs_bootstrap.png"
    pdf_path = FIGURES_DIR / "process2_stability_direct_vs_bootstrap.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
