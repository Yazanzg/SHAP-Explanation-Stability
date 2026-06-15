"""Smoke test: run the full pipeline end-to-end on the 10-row synthetic dataset.

This proves the config-driven pipeline works without the controlled PROCESS-2
data. It is deliberately fast (target < 2 minutes). Run with the synthetic
config:

    # PowerShell
    $env:THESIS_CONFIG = "examples/synthetic/config.yaml"
    python scripts/smoke_test.py

Stages exercised: load cohort CSV -> clean transcripts -> extract the 13
configured features -> fit each model family on train -> evaluate on the fixed
dev split -> compute SHAP and reduce to a mean-|SHAP| vector of length 13.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.config import (
    COL_LABEL,
    COL_PARTICIPANT_ID,
    COL_SPLIT,
    FEATURE_COLUMNS,
    MODELS,
    TEST_SPLIT_VALUE,
    TRAIN_SPLIT_VALUE,
)
from src.data_loading import load_dataset_table
from src.explainability import compute_shap_values
from src.feature_extraction import build_feature_matrix
from src.modeling import build_estimator

TIME_BUDGET_SECONDS = 120.0


def main() -> None:
    t0 = time.perf_counter()
    print(f"[smoke] config: {config.CONFIG_PATH}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load cohort table (dataset-agnostic, config-driven column names)
    table = load_dataset_table()
    print(f"[smoke] loaded {len(table)} participants")

    # 2) Extract the 13 configured features from raw transcripts
    feats = build_feature_matrix(
        table,
        text_column="transcript",
        id_column=COL_PARTICIPANT_ID,
        label_column=COL_LABEL,
        split_column=COL_SPLIT,
    )
    n_nan = int(feats[list(FEATURE_COLUMNS)].isna().sum().sum())
    assert len(feats) == len(table), "feature matrix row count mismatch"
    assert list(FEATURE_COLUMNS) == [c for c in FEATURE_COLUMNS if c in feats.columns], (
        "feature columns missing"
    )
    assert n_nan == 0, f"unexpected NaNs in synthetic feature matrix: {n_nan}"

    config.FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(config.FEATURES_CSV, index=False)
    print(f"[smoke] wrote features -> {config.FEATURES_CSV} ({len(FEATURE_COLUMNS)} features)")

    # 3) Train each model family, evaluate on the fixed dev split, compute SHAP
    train = feats[feats[COL_SPLIT] == TRAIN_SPLIT_VALUE].reset_index(drop=True)
    dev = feats[feats[COL_SPLIT] == TEST_SPLIT_VALUE].reset_index(drop=True)
    assert len(train) > 0 and len(dev) > 0, "empty train or dev split"

    X_train = train[list(FEATURE_COLUMNS)]
    y_train = train[COL_LABEL].to_numpy(dtype=int)
    X_dev = dev[list(FEATURE_COLUMNS)]
    y_dev = dev[COL_LABEL].to_numpy(dtype=int)

    for model_name in MODELS:
        pipe = build_estimator(model_name)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_dev)[:, 1]
        pred = pipe.predict(X_dev)
        auroc = (
            float(roc_auc_score(y_dev, proba)) if len(np.unique(y_dev)) > 1 else float("nan")
        )
        acc = float(accuracy_score(y_dev, pred))

        shap_long = compute_shap_values(model_name, pipe, dev, variant="smoke")
        mean_abs = (
            shap_long.groupby("feature_name")["shap_value"]
            .apply(lambda s: float(np.mean(np.abs(s))))
            .reindex(FEATURE_COLUMNS)
        )
        assert len(mean_abs) == len(FEATURE_COLUMNS), "mean-|SHAP| vector wrong length"
        print(
            f"[smoke] {model_name:<20} auroc={auroc:.3f} acc={acc:.3f} "
            f"shap_vec_len={len(mean_abs)}"
        )

    elapsed = time.perf_counter() - t0
    print(f"[smoke] completed in {elapsed:.1f}s (budget {TIME_BUDGET_SECONDS:.0f}s)")
    if elapsed > TIME_BUDGET_SECONDS:
        raise SystemExit(f"Smoke test exceeded time budget: {elapsed:.1f}s")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
