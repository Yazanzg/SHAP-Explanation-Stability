#!/usr/bin/env python3
"""
End-to-end thesis pipeline: PROCESS-2 CTD features, models, SHAP, stability, figures.

Run from project root:
    python run_pipeline.py

Uses predefined Train/Test splits from PROCESS-2 ``metadata.csv`` when
``THESIS_DATASET=PROCESS2`` (default). Cross-validation, hyperparameter search,
shuffled-label checks, and explanation stability run on **training data only**.
Held-out **test** participants are never used for tuning or SHAP background fitting.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.config import (
    DATASET_NAME,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    OUTPUT_BASELINE,
    OUTPUT_FEATURES,
    OUTPUT_OPTIMIZED,
    OUTPUT_SHAP_BASELINE,
    OUTPUT_SHAP_OPTIMIZED,
    OUTPUT_SHUFFLED,
    OUTPUT_STABILITY,
    is_process2,
)
from src.data_loading import (
    load_cohort_table,
    print_directory_tree,
    summarize_cohort,
    train_test_masks,
    validate_paths,
    validate_process2,
)
from src.explainability import compute_shap_suite
from src.feature_extraction import extract_features_table
from src.modeling import get_named_baselines, run_baseline_suite, run_shuffled_label_suite
from src.optimization import run_optimization_suite
from src.preprocessing import build_cleaned_corpus, run_quality_checks
from src.stability import (
    compare_baseline_vs_optimized_global,
    composite_stability_score,
    run_stability_analysis,
)
from src.utils import ensure_directories, setup_logging
from src.visualization import (
    plot_performance_imbalance_panel,
    plot_stability_comparison,
    plot_tradeoff_scatter,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ensure_directories(config.OUTPUT_DIR, FIGURES_DIR, config.NOTEBOOKS_DIR)

    logger.info("Dataset mode: %s  |  root: %s", DATASET_NAME, config.DATA_DIR)

    print_directory_tree(config.DATA_DIR, max_depth=2, max_files=60)
    print("\n", validate_paths())

    cohort = load_cohort_table()
    if is_process2():
        validate_process2(meta=cohort)
        train_m, test_m = train_test_masks(cohort)
        has_test = True
    else:
        train_m = pd.Series(True, index=cohort.index)
        test_m = pd.Series(False, index=cohort.index)
        has_test = False
        logger.warning(
            "Legacy dataset mode: no predefined Test split; CV/SHAP/stability use all samples."
        )

    summarize_cohort(cohort, full_meta=cohort)

    corpus = build_cleaned_corpus(cohort["participant_id"].astype(str).tolist())
    run_quality_checks(corpus)

    feats = extract_features_table(corpus)
    table = feats.merge(
        cohort[["participant_id", "diagnosis", "diagnosis_label", "split"]],
        on="participant_id",
        how="inner",
    )
    if table["diagnosis_label"].isna().any():
        raise ValueError("Missing diagnosis labels after merge.")

    save_cols = [
        "participant_id",
        "diagnosis",
        "diagnosis_label",
        "split",
        "cleaned_transcript",
        "word_count",
        *FEATURE_COLUMNS,
    ]
    table[save_cols].to_csv(OUTPUT_FEATURES, index=False)
    logger.info("Saved features -> %s", OUTPUT_FEATURES)

    X_train = table.loc[train_m].copy()
    y_train = X_train["diagnosis_label"].values.astype(int)
    X_test = table.loc[test_m].copy() if has_test else None
    y_test = X_test["diagnosis_label"].values.astype(int) if has_test else None

    if has_test:
        logger.info(
            "Train n=%s  |  Test n=%s (test held out from tuning/SHAP/stability)",
            len(X_train),
            len(X_test),
        )

    baseline_df = run_baseline_suite(
        X_train, y_train, X_test=X_test, y_test=y_test if has_test else None
    )
    baseline_df.to_csv(OUTPUT_BASELINE, index=False)
    logger.info("Saved baseline results -> %s", OUTPUT_BASELINE)

    shuf_df = run_shuffled_label_suite(X_train, y_train)
    shuf_df.to_csv(OUTPUT_SHUFFLED, index=False)
    logger.info("Saved shuffled-label results -> %s", OUTPUT_SHUFFLED)

    opt_df, opt_models = run_optimization_suite(
        X_train, y_train, X_test=X_test, y_test=y_test if has_test else None
    )
    opt_df.to_csv(OUTPUT_OPTIMIZED, index=False)
    logger.info("Saved optimized results -> %s", OUTPUT_OPTIMIZED)

    # SHAP and stability: training set only (no test leakage)
    baseline_fitted: dict = {}
    for name, tmpl in get_named_baselines().items():
        m = clone(tmpl)
        m.fit(X_train[list(FEATURE_COLUMNS)], y_train)
        baseline_fitted[name] = m

    shap_b = compute_shap_suite(baseline_fitted, X_train, variant="baseline_train")
    shap_b.to_csv(OUTPUT_SHAP_BASELINE, index=False)
    logger.info("Saved SHAP baseline (train) -> %s", OUTPUT_SHAP_BASELINE)

    shap_o = compute_shap_suite(opt_models, X_train, variant="optimized_train")
    shap_o.to_csv(OUTPUT_SHAP_OPTIMIZED, index=False)
    logger.info("Saved SHAP optimized (train) -> %s", OUTPUT_SHAP_OPTIMIZED)

    skip_svm = os.environ.get("THESIS_SKIP_SVM_STABILITY", "1") == "1"
    stab_parts = [run_stability_analysis(X_train, y_train, skip_svm_shap=skip_svm)]
    stab_parts.append(compare_baseline_vs_optimized_global(shap_b, shap_o))
    stab_df = pd.concat(stab_parts, ignore_index=True)
    stab_df.to_csv(OUTPUT_STABILITY, index=False)
    logger.info("Saved stability -> %s", OUTPUT_STABILITY)

    stab_core = stab_df[
        stab_df["metric"].isin(
            [
                "spearman_rank_importance",
                "jaccard_top5_features",
                "mean_cv_shap_magnitude_across_features",
            ]
        )
    ]
    stab_sum = composite_stability_score(stab_core)

    plot_performance_imbalance_panel(
        baseline_df,
        opt_df,
        FIGURES_DIR / "performance_comparison.png",
    )
    plot_stability_comparison(
        stab_df[stab_df["metric"] == "spearman_rank_importance"],
        FIGURES_DIR / "stability_comparison.png",
    )

    perf_for_tradeoff = _performance_for_tradeoff(baseline_df, opt_df)
    plot_tradeoff_scatter(
        perf_for_tradeoff,
        stab_sum,
        FIGURES_DIR / "tradeoff_auroc_stability.png",
    )

    logger.info("Pipeline complete. Outputs under %s", config.OUTPUT_DIR)


def _performance_for_tradeoff(baseline_df: pd.DataFrame, opt_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer held-out test metrics; fall back to train CV for trade-off plots."""
    parts = []
    for df, kind in [(baseline_df, "baseline"), (opt_df, "optimized")]:
        sub = df.copy()
        if "eval_scope" in sub.columns:
            test_rows = sub[sub["eval_scope"] == "test"]
            sub = test_rows if len(test_rows) else sub[sub["eval_scope"] == "train_cv"]
        metric_col = "roc_auc_mean" if "roc_auc_mean" in sub.columns else "roc_auc"
        parts.append(sub[["model", metric_col]].rename(columns={metric_col: "roc_auc_mean"}).assign(kind=kind))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    main()
