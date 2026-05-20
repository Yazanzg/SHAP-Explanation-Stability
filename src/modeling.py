"""Baseline classifiers, evaluation metrics, stratified CV, and held-out test scoring."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.config import CV_SPLITS, FEATURE_COLUMNS, INCLUDE_BALANCED_VARIANTS, RANDOM_SEED

logger = logging.getLogger(__name__)


def make_preprocess_numeric() -> ColumnTransformer:
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), list(FEATURE_COLUMNS))],
        remainder="drop",
    )


def baseline_logistic_pipeline(*, balanced: bool = False) -> Pipeline:
    cw: str | None = "balanced" if balanced else None
    return Pipeline(
        [
            ("prep", make_preprocess_numeric()),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=10000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                    class_weight=cw,
                ),
            ),
        ]
    )


def baseline_rf_pipeline(*, balanced: bool = False) -> Pipeline:
    cw: str | None = "balanced" if balanced else None
    return Pipeline(
        [
            ("prep", make_preprocess_numeric()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_SEED,
                    class_weight=cw,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def baseline_svm_pipeline(*, balanced: bool = False) -> Pipeline:
    cw: str | None = "balanced" if balanced else None
    return Pipeline(
        [
            ("prep", make_preprocess_numeric()),
            ("scale", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=RANDOM_SEED,
                    class_weight=cw,
                ),
            ),
        ]
    )


def get_named_baselines() -> dict[str, Pipeline]:
    """Primary LR / RF / SVM; optional ``*_balanced`` variants when configured."""
    models: dict[str, Pipeline] = {
        "logistic_regression": baseline_logistic_pipeline(balanced=False),
        "random_forest": baseline_rf_pipeline(balanced=False),
        "svm_rbf": baseline_svm_pipeline(balanced=False),
    }
    if INCLUDE_BALANCED_VARIANTS:
        models.update(
            {
                "logistic_regression_balanced": baseline_logistic_pipeline(balanced=True),
                "random_forest_balanced": baseline_rf_pipeline(balanced=True),
                "svm_rbf_balanced": baseline_svm_pipeline(balanced=True),
            }
        )
    return models


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba_pos: np.ndarray) -> dict[str, float]:
    zm = {"zero_division": 0}
    m: dict[str, float] = {
        "roc_auc": _safe_roc_auc(y_true, y_proba_pos),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, **zm)),
        "precision": float(precision_score(y_true, y_pred, **zm)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, **zm)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "average_precision": _safe_average_precision(y_true, y_proba_pos),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, **zm)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, **zm)),
        "recall_negative": float(recall_score(y_true, y_pred, pos_label=0, **zm)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["tn"] = float(tn)
    m["fp"] = float(fp)
    m["fn"] = float(fn)
    m["tp"] = float(tp)
    return m


def _metrics_to_summary(m: dict[str, float], *, std: float = 0.0) -> dict[str, float]:
    return {f"{k}_mean": float(m[k]) for k in m} | {f"{k}_std": std for k in m}


def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cv_splits: int = CV_SPLITS,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Stratified K-fold CV on ``X`` (training set only; no test leakage)."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    Xmat = X[list(FEATURE_COLUMNS)]
    fold_rows: list[dict[str, float]] = []
    for train_idx, test_idx in cv.split(Xmat, y):
        est = clone(pipeline)
        est.fit(Xmat.iloc[train_idx], y[train_idx])
        y_pred = est.predict(Xmat.iloc[test_idx])
        y_proba = est.predict_proba(Xmat.iloc[test_idx])[:, 1]
        fold_rows.append(_fold_metrics(y[test_idx], y_pred, y_proba))

    keys = list(fold_rows[0].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([fr[k] for fr in fold_rows], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
    return out


def evaluate_pipeline_holdout(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit on train, evaluate once on held-out test (predefined split)."""
    est = clone(pipeline)
    est.fit(X_train[list(FEATURE_COLUMNS)], y_train)
    y_pred = est.predict(X_test[list(FEATURE_COLUMNS)])
    y_proba = est.predict_proba(X_test[list(FEATURE_COLUMNS)])[:, 1]
    m = _fold_metrics(y_test, y_pred, y_proba)
    return _metrics_to_summary(m, std=0.0)


def run_baseline_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, pipe in get_named_baselines().items():
        logger.info("Cross-validating baseline (train): %s", name)
        m_cv = evaluate_pipeline_cv(pipe, X_train, y_train)
        rows.append(
            {
                "model": name,
                "variant": "baseline",
                "eval_scope": "train_cv",
                **m_cv,
            }
        )
        if X_test is not None and y_test is not None:
            logger.info("Held-out test evaluation (baseline): %s", name)
            m_te = evaluate_pipeline_holdout(pipe, X_train, y_train, X_test, y_test)
            rows.append(
                {
                    "model": name,
                    "variant": "baseline",
                    "eval_scope": "test",
                    **m_te,
                }
            )
    return pd.DataFrame(rows)


def run_shuffled_label_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """Sanity check on training data only; labels permuted → chance-level ROC-AUC."""
    rng = np.random.default_rng(seed=RANDOM_SEED)
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)
    rows = []
    for name, pipe in get_named_baselines().items():
        logger.info("Shuffled-label CV (train): %s", name)
        m = evaluate_pipeline_cv(pipe, X_train, y_shuf)
        rows.append(
            {
                "model": name,
                "variant": "shuffled_labels",
                "eval_scope": "shuffled_train_cv",
                **m,
            }
        )
    return pd.DataFrame(rows)
