"""Explanation stability: CV / bootstrap agreement and baseline vs optimized."""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import CV_SPLITS, FEATURE_COLUMNS, RANDOM_SEED, STABILITY_BOOTSTRAPS
from src.explainability import compute_shap_values
from src.modeling import get_named_baselines

logger = logging.getLogger(__name__)


def _mean_abs_vector_from_long(shap_long: pd.DataFrame, model: str) -> pd.Series:
    sub = shap_long[shap_long["model"] == model]
    g = sub.groupby("feature_name")["shap_value"].apply(lambda s: np.mean(np.abs(s)))
    return g.reindex(FEATURE_COLUMNS).fillna(0.0)


def _spearman_between_importance(a: pd.Series, b: pd.Series) -> float:
    r, _ = spearmanr(a.values, b.values)
    return float(r) if not np.isnan(r) else 0.0


def _jaccard_topk(a: pd.Series, b: pd.Series, k: int = 5) -> float:
    ta = set(a.nlargest(k).index)
    tb = set(b.nlargest(k).index)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def _cv_of_series(vals: np.ndarray) -> float:
    m = float(np.mean(vals))
    if m == 0:
        return float("nan")
    return float(np.std(vals) / m)


def run_stability_analysis(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    skip_svm_shap: bool = False,
) -> pd.DataFrame:
    """
    For each baseline model, refit on CV folds (and bootstrap resamples),
    compute SHAP on the **training subset** of that run, derive global mean |SHAP|,
    then measure Spearman rank agreement, Jaccard(top-5), and CV of SHAP magnitudes.
    """
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict[str, object]] = []

    baselines = get_named_baselines()

    for model_name, template in baselines.items():
        if skip_svm_shap and model_name.startswith("svm"):
            logger.warning(
                "Skipping SVM SHAP in stability for %s (set skip_svm_shap=False for full run).",
                model_name,
            )
            continue

        fold_vecs: list[pd.Series] = []
        for fold_idx, (tr, _) in enumerate(cv.split(X, y)):
            pipe = clone(template)
            pipe.fit(X.iloc[tr][list(FEATURE_COLUMNS)], y[tr])
            X_tr = X.iloc[tr].copy()
            sh_long = compute_shap_values(model_name, pipe, X_tr, variant=f"baseline_cv_fold_{fold_idx}")
            fold_vecs.append(_mean_abs_vector_from_long(sh_long, model_name))

        records.extend(
            _aggregate_run_metrics(
                model_name, "baseline_cv", fold_vecs, extra_tag="cv_folds"
            )
        )

        # Bootstrap on full dataset indices
        boot_vecs: list[pd.Series] = []
        for b in range(STABILITY_BOOTSTRAPS):
            idx = rng.integers(0, len(X), size=len(X))
            Xb = X.iloc[idx].reset_index(drop=True)
            yb = y[idx]
            pipe = clone(template)
            pipe.fit(Xb[list(FEATURE_COLUMNS)], yb)
            sh_long = compute_shap_values(model_name, pipe, Xb, variant=f"baseline_bootstrap_{b}")
            boot_vecs.append(_mean_abs_vector_from_long(sh_long, model_name))

        records.extend(
            _aggregate_run_metrics(
                model_name, "baseline_bootstrap", boot_vecs, extra_tag="bootstrap"
            )
        )

    return pd.DataFrame.from_records(records)


def _aggregate_run_metrics(
    model_name: str,
    variant_scope: str,
    vectors: list[pd.Series],
    *,
    extra_tag: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if len(vectors) < 2:
        return out
    sp = []
    for vi, vj in combinations(range(len(vectors)), 2):
        sp.append(_spearman_between_importance(vectors[vi], vectors[vj]))
    jac = []
    for vi, vj in combinations(range(len(vectors)), 2):
        jac.append(_jaccard_topk(vectors[vi], vectors[vj], k=5))

    mat = np.stack([v.values for v in vectors], axis=0)
    feat_cv = np.nanmean([_cv_of_series(mat[:, j]) for j in range(mat.shape[1])])

    out.append(
        {
            "metric": "spearman_rank_importance",
            "scope": extra_tag,
            "model": model_name,
            "variant": variant_scope,
            "value": float(np.mean(sp)),
        }
    )
    out.append(
        {
            "metric": "jaccard_top5_features",
            "scope": extra_tag,
            "model": model_name,
            "variant": variant_scope,
            "value": float(np.mean(jac)),
        }
    )
    out.append(
        {
            "metric": "mean_cv_shap_magnitude_across_features",
            "scope": extra_tag,
            "model": model_name,
            "variant": variant_scope,
            "value": float(feat_cv),
        }
    )
    return out


def compare_baseline_vs_optimized_global(
    shap_baseline: pd.DataFrame,
    shap_optimized: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman correlation between global mean |SHAP| profiles."""
    rows = []
    models = sorted(set(shap_baseline["model"]) & set(shap_optimized["model"]))
    for m in models:
        vb = _mean_abs_vector_from_long(shap_baseline, m)
        vo = _mean_abs_vector_from_long(shap_optimized, m)
        rows.append(
            {
                "metric": "spearman_baseline_vs_optimized_global_importance",
                "scope": "baseline_vs_optimized",
                "model": m,
                "variant": "comparison",
                "value": _spearman_between_importance(vb, vo),
            }
        )
        rows.append(
            {
                "metric": "jaccard_top5_baseline_vs_optimized",
                "scope": "baseline_vs_optimized",
                "model": m,
                "variant": "comparison",
                "value": _jaccard_topk(vb, vo, k=5),
            }
        )
    return pd.DataFrame(rows)


def composite_stability_score(stab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Single scalar per model for trade-off plots: average of normalized core metrics.
    """
    rows = []
    for model, g in stab_df.groupby("model"):
        spearman = float(
            np.nanmean(g.loc[g["metric"] == "spearman_rank_importance", "value"])
        )
        jaccard = float(
            np.nanmean(g.loc[g["metric"] == "jaccard_top5_features", "value"])
        )
        cv_mag = float(
            np.nanmean(
                g.loc[
                    g["metric"] == "mean_cv_shap_magnitude_across_features", "value"
                ]
            )
        )
        cv_score = 1.0 / (1.0 + float(cv_mag)) if not np.isnan(cv_mag) else 0.5
        score = float(np.clip((spearman + jaccard + cv_score) / 3.0, 0.0, 1.0))
        rows.append({"model": model, "stability_composite": score})
    return pd.DataFrame(rows)
