"""Model construction and evaluation for the training-size stability pipeline.

Models are built from ``config.yaml`` with fixed, pre-specified hyperparameters.
No tuning, GridSearchCV, or performance-based selection happens anywhere. There
is no imputation step: the feature matrix has no missing values (see the
imputation audit), so imputation would be statistically inappropriate.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import MODELS, RANDOM_SEED

logger = logging.getLogger(__name__)

# Map the estimator name in config.yaml to its scikit-learn class.
_ESTIMATOR_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
}


def build_estimator(model_name: str) -> Pipeline:
    """Construct a model pipeline from config.yaml (fixed settings, no imputation).

    The pipeline is ``[scale?, clf]`` with the classifier step named ``clf`` and
    an optional ``StandardScaler`` step named ``scale`` when the config sets
    ``scale_features: true``. Hyperparameters are taken verbatim from config; a
    fixed ``random_state`` is injected for reproducibility. There is no
    SimpleImputer: the feature matrix has no missing values (see the imputation
    audit), so imputation would be inappropriate.
    """
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Configured: {list(MODELS)}")
    spec = MODELS[model_name]
    est_name = spec["estimator"]
    if est_name not in _ESTIMATOR_REGISTRY:
        raise KeyError(f"Unsupported estimator '{est_name}' for model '{model_name}'.")
    cls = _ESTIMATOR_REGISTRY[est_name]
    params = dict(spec.get("hyperparameters", {}))
    params.setdefault("random_state", RANDOM_SEED)
    steps: list[tuple[str, Any]] = []
    if spec.get("scale_features", False):
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", cls(**params)))
    return Pipeline(steps)


def evaluate_on_split(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Score a fitted pipeline on a fixed evaluation split.

    Returns AUROC, F1, accuracy, precision, recall. AUROC uses the positive-class
    probability; the others use the hard prediction.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    auroc = (
        float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return {
        "auroc": auroc,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
    }
