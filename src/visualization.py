"""Figures for thesis: performance, stability, and performance–stability trade-offs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATASET_NAME, FIGURES_DIR

logger = logging.getLogger(__name__)


def _ensure_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _select_eval_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer held-out test rows; otherwise train CV."""
    if "eval_scope" not in df.columns:
        return df
    test_df = df[df["eval_scope"] == "test"]
    if len(test_df):
        return test_df
    return df[df["eval_scope"] == "train_cv"]


def plot_performance_comparison(
    baseline: pd.DataFrame,
    optimized: pd.DataFrame,
    out_path: Path,
) -> None:
    _ensure_dir()
    b = _select_eval_scope(baseline)[["model", "roc_auc_mean"]].copy()
    b["kind"] = "baseline"
    o = _select_eval_scope(optimized)[["model", "roc_auc_mean"]].copy()
    o["kind"] = "optimized"
    df = pd.concat([b, o], ignore_index=True)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="roc_auc_mean", hue="kind")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Wrote %s", out_path)


def plot_performance_imbalance_panel(
    baseline: pd.DataFrame,
    optimized: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Primary performance figure (PROCESS-2): ROC-AUC, PR-AUC, balanced accuracy, F1.
    """
    _ensure_dir()
    b = _select_eval_scope(baseline)
    o = _select_eval_scope(optimized)
    metrics: list[tuple[str, str]] = [
        ("roc_auc_mean", "ROC-AUC"),
        ("average_precision_mean", "PR-AUC / average precision"),
        ("balanced_accuracy_mean", "Balanced accuracy"),
        ("f1_mean", "F1-score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, ylab) in zip(axes.flat, metrics):
        if col not in b.columns:
            continue
        bb = b[["model", col]].copy()
        bb["kind"] = "baseline"
        oo = o[["model", col]].copy()
        oo["kind"] = "optimized"
        df = pd.concat([bb, oo], ignore_index=True)
        sns.barplot(data=df, x="model", y=col, hue="kind", ax=ax)
        ax.set_ylabel(ylab)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=22)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, title="")
    scope_note = "test" if "eval_scope" in baseline.columns and (baseline["eval_scope"] == "test").any() else "train CV"
    fig.suptitle(
        f"{DATASET_NAME} CTD — performance ({scope_note} split)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", out_path)


def plot_stability_comparison(stab: pd.DataFrame, out_path: Path) -> None:
    _ensure_dir()
    metrics = [
        "spearman_rank_importance",
        "jaccard_top5_features",
        "mean_cv_shap_magnitude_across_features",
    ]
    sub = stab[stab["metric"].isin(metrics)].copy()
    if sub.empty:
        sub = stab[stab["metric"] == "spearman_rank_importance"].copy()
    fig, axes = plt.subplots(1, min(3, len(metrics)), figsize=(5 * min(3, len(metrics)), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, metric in zip(axes.flat, metrics):
        msub = sub[sub["metric"] == metric]
        if msub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=msub, x="model", y="value", hue="scope", ax=ax)
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle(f"{DATASET_NAME} — explanation stability (train only)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", out_path)


def plot_tradeoff_scatter(
    perf: pd.DataFrame,
    stab_summary: pd.DataFrame,
    out_path: Path,
    *,
    x_metric: str = "roc_auc_mean",
) -> None:
    """Scatter: performance (ROC-AUC or PR-AUC) vs stability composite."""
    _ensure_dir()
    m = perf.merge(stab_summary, on="model", how="inner")
    if x_metric not in m.columns and "average_precision_mean" in m.columns:
        x_metric = "average_precision_mean"
    plt.figure(figsize=(6, 5))
    rng = np.random.default_rng(42)
    for kind, g in m.groupby("kind"):
        yj = g["stability_composite"].values + rng.normal(0, 0.015, size=len(g))
        plt.scatter(g[x_metric], yj, label=kind, alpha=0.85)
        for (_, r), yy in zip(g.iterrows(), yj):
            plt.annotate(r["model"], (r[x_metric], yy), fontsize=7)
    plt.xlabel(x_metric.replace("_mean", "").replace("_", " ").upper())
    plt.ylabel("Explanation stability (composite)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{DATASET_NAME} — performance vs explanation stability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Wrote %s", out_path)


def build_tradeoff_table(
    baseline: pd.DataFrame,
    optimized: pd.DataFrame,
    stab_summary: pd.DataFrame,
) -> pd.DataFrame:
    b = _select_eval_scope(baseline)[["model", "roc_auc_mean"]].copy()
    b["kind"] = "baseline"
    o = _select_eval_scope(optimized)[["model", "roc_auc_mean"]].copy()
    o["kind"] = "optimized"
    perf = pd.concat([b, o], ignore_index=True)
    return perf.merge(stab_summary, on="model", how="left")
