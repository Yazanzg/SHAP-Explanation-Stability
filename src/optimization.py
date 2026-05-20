"""Hyperparameter search on training data only; held-out test evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.config import CV_SPLITS, FEATURE_COLUMNS, INCLUDE_BALANCED_VARIANTS, RANDOM_SEED
from src.modeling import (
    evaluate_pipeline_cv,
    evaluate_pipeline_holdout,
    make_preprocess_numeric,
)

logger = logging.getLogger(__name__)


def _rfe_estimator() -> LogisticRegression:
    return LogisticRegression(max_iter=8000, random_state=RANDOM_SEED, solver="lbfgs")


def optimized_logistic_pipeline(*, balanced: bool) -> Pipeline:
    cw: str | None = "balanced" if balanced else None
    rfe = RFE(estimator=_rfe_estimator(), step=1)
    return Pipeline(
        [
            ("prep", make_preprocess_numeric()),
            ("scale", StandardScaler()),
            ("rfe", rfe),
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


def logistic_param_grid() -> dict[str, list[Any]]:
    nfeat = len(FEATURE_COLUMNS)
    n_list = sorted({4, 6, min(8, nfeat)})
    return {
        "rfe__n_features_to_select": n_list,
        "clf__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    }


def optimized_svm_pipeline(*, balanced: bool) -> Pipeline:
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


def svm_param_grid() -> dict[str, list[Any]]:
    return {
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    }


def optimized_rf_pipeline(*, balanced: bool) -> Pipeline:
    cw: str | None = "balanced" if balanced else None
    return Pipeline(
        [
            ("prep", make_preprocess_numeric()),
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_SEED,
                    class_weight=cw,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def rf_param_grid() -> dict[str, list[Any]]:
    return {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 4, 6, 8],
        "clf__min_samples_split": [2, 4, 6],
        "clf__min_samples_leaf": [1, 2, 4],
    }


def _optimization_jobs() -> list[tuple[str, Pipeline, dict[str, list[Any]]]]:
    jobs: list[tuple[str, Pipeline, dict[str, list[Any]]]] = [
        ("logistic_regression", optimized_logistic_pipeline(balanced=False), logistic_param_grid()),
        ("svm_rbf", optimized_svm_pipeline(balanced=False), svm_param_grid()),
        ("random_forest", optimized_rf_pipeline(balanced=False), rf_param_grid()),
    ]
    if INCLUDE_BALANCED_VARIANTS:
        jobs.extend(
            [
                (
                    "logistic_regression_balanced",
                    optimized_logistic_pipeline(balanced=True),
                    logistic_param_grid(),
                ),
                ("svm_rbf_balanced", optimized_svm_pipeline(balanced=True), svm_param_grid()),
                (
                    "random_forest_balanced",
                    optimized_rf_pipeline(balanced=True),
                    rf_param_grid(),
                ),
            ]
        )
    return jobs


def grid_search_best(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Pipeline, dict[str, Any]]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train[list(FEATURE_COLUMNS)], y_train)
    logger.info("Best params (%s): %s", pipeline.steps[-1][0], gs.best_params_)
    return gs.best_estimator_, dict(gs.best_params_)


def run_optimization_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """
    GridSearchCV on training data only; report train CV metrics and optional test metrics.
    """
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for model_key, tmpl, grid in _optimization_jobs():
        logger.info("GridSearchCV (train only): %s", model_key)
        best, bp = grid_search_best(tmpl, grid, X_train, y_train)
        fitted[model_key] = best
        m_cv = evaluate_pipeline_cv(best, X_train, y_train)
        rows.append(
            {
                "model": model_key,
                "variant": "optimized",
                "eval_scope": "train_cv",
                **m_cv,
                "best_params": str(bp),
            }
        )
        if X_test is not None and y_test is not None:
            m_te = evaluate_pipeline_holdout(best, X_train, y_train, X_test, y_test)
            rows.append(
                {
                    "model": model_key,
                    "variant": "optimized",
                    "eval_scope": "test",
                    **m_te,
                    "best_params": str(bp),
                }
            )

    return pd.DataFrame(rows), fitted
