"""SHAP explainers tailored to sklearn pipelines (LR / RF / SVM)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    KERNEL_SHAP_NSAMPLES,
    RANDOM_SEED,
    SHAP_BACKGROUND_SIZE,
)

logger = logging.getLogger(__name__)


def _transform_until_classifier(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply all steps before the final classifier; return design matrix."""
    X_iter = X[list(FEATURE_COLUMNS)]
    for _, step in pipeline.steps[:-1]:
        X_iter = step.transform(X_iter)
    return np.asarray(X_iter)


def _expand_to_full(
    shap_sub: np.ndarray, support: np.ndarray | None
) -> np.ndarray:
    """Map reduced-feature SHAP values back to full FEATURE_COLUMNS layout."""
    n_samples = shap_sub.shape[0]
    full = np.zeros((n_samples, len(FEATURE_COLUMNS)))
    if support is None:
        full[:, :] = shap_sub
        return full
    idx = np.flatnonzero(support)
    for j, col in enumerate(idx):
        full[:, col] = shap_sub[:, j]
    return full


def compute_shap_values(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    """
    Compute SHAP values for each participant (row) and stack as a long table.

    Uses LinearExplainer (LR), TreeExplainer (RF), KernelExplainer (SVM RBF).
    """
    X_feat = X[list(FEATURE_COLUMNS)]
    bg = X_feat.iloc[
        np.random.default_rng(RANDOM_SEED).choice(
            len(X_feat), size=min(SHAP_BACKGROUND_SIZE, len(X_feat)), replace=False
        )
    ]

    clf = pipeline.named_steps["clf"]
    X_model = _transform_until_classifier(pipeline, X_feat)
    X_bg = _transform_until_classifier(pipeline, bg)

    support = None
    if "rfe" in pipeline.named_steps:
        support = pipeline.named_steps["rfe"].support_

    # Map extended model keys (e.g. ``*_balanced``) to explainer families.
    if model_name.startswith("logistic_regression"):
        masker = shap.maskers.Independent(X_bg)
        explainer = shap.LinearExplainer(clf, masker)
        sv = explainer(X_model)
        vals = np.array(sv.values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    elif model_name.startswith("random_forest"):
        explainer = shap.TreeExplainer(clf, data=X_bg)
        sv = explainer(X_model)
        vals = np.array(sv.values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    elif model_name.startswith("svm"):
        # --- SVM + KernelSHAP (SHAP version compatibility) ---
        # SHAP 0.45+ exposes ``KernelExplainer.__call__(X, l1_reg=..., silent=...)`` only;
        # ``nsamples`` is **not** forwarded through ``__call__`` (it was removed from the
        # call signature). Sampling budget belongs to ``shap_values(X, nsamples=..., ...)``,
        # which passes ``nsamples`` into the internal ``explain()`` path (see
        # ``shap.explainers._kernel.KernelExplainer.shap_values`` docstring).
        # Older SHAP builds accepted ``nsamples`` on ``__call__``; we try ``shap_values``
        # first, then fall back for compatibility.
        bg_raw = shap.sample(
            X_feat,
            min(SHAP_BACKGROUND_SIZE, len(X_feat)),
            random_state=RANDOM_SEED,
        )

        def f(d: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(d, columns=list(FEATURE_COLUMNS))
            return pipeline.predict_proba(frame)

        x_arr = X_feat.values.astype(float)
        support = None
        try:
            explainer = shap.KernelExplainer(f, bg_raw.values.astype(float))
            try:
                vals = explainer.shap_values(
                    x_arr,
                    nsamples=KERNEL_SHAP_NSAMPLES,
                    silent=True,
                )
            except TypeError:
                # Legacy SHAP: try __call__ with nsamples; if unsupported, call without nsamples.
                try:
                    out = explainer(
                        x_arr,
                        nsamples=KERNEL_SHAP_NSAMPLES,
                        silent=True,
                    )
                    vals = np.asarray(out.values) if hasattr(out, "values") else np.asarray(out)
                except TypeError:
                    out = explainer(x_arr, silent=True)
                    vals = np.asarray(out.values) if hasattr(out, "values") else np.asarray(out)
        except Exception as exc:
            logger.warning(
                "SVM KernelSHAP failed (%s: %s). Continuing with NaN SHAP values for %s; "
                "other model families are unchanged.",
                type(exc).__name__,
                exc,
                model_name,
            )
            vals = np.full((len(X_feat), len(FEATURE_COLUMNS)), np.nan, dtype=float)

        if isinstance(vals, list):
            vals = vals[1]
        vals = np.asarray(vals, dtype=float)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown model_name for SHAP: {model_name}")

    if not model_name.startswith("svm"):
        vals = _expand_to_full(vals, support)

    rows = []
    pids = X["participant_id"].values if "participant_id" in X.columns else np.arange(len(X))
    for i, pid in enumerate(pids):
        for j, fname in enumerate(FEATURE_COLUMNS):
            rows.append(
                {
                    "participant_id": str(pid),
                    "feature_name": fname,
                    "shap_value": float(vals[i, j]),
                    "model": model_name,
                    "variant": variant,
                }
            )
    return pd.DataFrame(rows)


def compute_shap_suite(
    fitted_models: dict[str, Pipeline],
    X: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    parts = []
    for name, pipe in fitted_models.items():
        logger.info("SHAP (%s): %s", variant, name)
        parts.append(compute_shap_values(name, pipe, X, variant=variant))
    return pd.concat(parts, ignore_index=True)
