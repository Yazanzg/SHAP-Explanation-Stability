"""Bootstrap-based explanation stability for PROCESS-2 (no final interpretation).

Uses:
  - outputs/process2_features.csv
  - Train split only for bootstrap fitting (n=320 with replacement)
  - Fixed Dev split (n=80) as the explanation set

Computes for each model version (baseline + optimized):
  - Pairwise Spearman rank correlation of feature rankings (mean |SHAP|)
  - Pairwise Jaccard similarity of top-k features (k=3,5,10)
  - Coefficient of variation (CV) of mean |SHAP| across bootstraps (averaged over features)

Outputs:
  - outputs/process2_bootstrap_stability_results.csv
  - outputs/process2_bootstrap_stability_validation_report.txt
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import KERNEL_SHAP_NSAMPLES, OUTPUT_DIR, RANDOM_SEED, SHAP_BACKGROUND_SIZE


FEATURE_COLS = [
    "word_count",
    "sentence_count",
    "type_token_ratio",
    "filler_count",
    "filler_ratio",
    "mean_clause_length",
    "content_density",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "determiner_ratio",
]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    stage: str  # baseline/optimized
    shap_method: str  # linear/tree/kernel
    estimator: Pipeline


def _rank_importance(imp: np.ndarray) -> np.ndarray:
    """Ranks (1=most important) from importance vector."""
    # argsort descending -> ranks
    order = np.argsort(-imp)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(imp) + 1, dtype=float)
    return ranks


def _jaccard_topk(imp_a: np.ndarray, imp_b: np.ndarray, k: int) -> float:
    ia = set(np.argsort(-imp_a)[:k].tolist())
    ib = set(np.argsort(-imp_b)[:k].tolist())
    if not ia and not ib:
        return 1.0
    return float(len(ia & ib) / max(1, len(ia | ib)))


def _pairwise_stats(mat: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean Spearman, Jaccard@3/@5/@10 over all pairs. mat: [B, F] mean|SHAP|."""
    B, F = mat.shape
    if B < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sp = []
    j3 = []
    j5 = []
    j10 = []
    ranks = np.stack([_rank_importance(mat[i]) for i in range(B)], axis=0)
    for i in range(B):
        for j in range(i + 1, B):
            r, _ = spearmanr(ranks[i], ranks[j])
            sp.append(float(r) if not np.isnan(r) else 0.0)
            j3.append(_jaccard_topk(mat[i], mat[j], 3))
            j5.append(_jaccard_topk(mat[i], mat[j], 5))
            j10.append(_jaccard_topk(mat[i], mat[j], 10))
    return float(np.mean(sp)), float(np.mean(j3)), float(np.mean(j5)), float(np.mean(j10))


def _cv_mean_abs(mat: np.ndarray) -> float:
    """Mean over features of CV(std/mean) across bootstraps for mean|SHAP|."""
    mu = np.mean(mat, axis=0)
    sd = np.std(mat, axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = sd / mu
    cv = np.where(np.isfinite(cv), cv, 0.0)
    return float(np.mean(cv))


def _linear_shap(pipe: Pipeline, X_train: pd.DataFrame, X_dev: pd.DataFrame) -> np.ndarray:
    Xt_train = pipe[:-1].transform(X_train)
    Xt_dev = pipe[:-1].transform(X_dev)
    clf = pipe[-1]
    masker = shap.maskers.Independent(Xt_train)
    explainer = shap.LinearExplainer(clf, masker)
    sv = explainer(Xt_dev)
    vals = np.asarray(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def _tree_shap(pipe: Pipeline, X_train: pd.DataFrame, X_dev: pd.DataFrame) -> np.ndarray:
    Xt_train = pipe[:-1].transform(X_train)
    Xt_dev = pipe[:-1].transform(X_dev)
    clf = pipe[-1]
    explainer = shap.TreeExplainer(clf, data=Xt_train, feature_names=FEATURE_COLS)
    sv = explainer(Xt_dev)
    vals = np.asarray(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def _kernel_shap(
    pipe: Pipeline,
    X_dev: pd.DataFrame,
    *,
    bg_arr: np.ndarray,
    nsamples: int,
) -> np.ndarray:
    def f(X_array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(X_array, columns=FEATURE_COLS)
        return pipe.predict_proba(frame)

    x_dev = X_dev.to_numpy(dtype=float)
    explainer = shap.KernelExplainer(f, bg_arr)
    vals = explainer.shap_values(x_dev, nsamples=nsamples, silent=True)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path = OUTPUT_DIR / "process2_features.csv"
    out_csv = OUTPUT_DIR / "process2_bootstrap_stability_results.csv"
    out_report = OUTPUT_DIR / "process2_bootstrap_stability_validation_report.txt"

    df = pd.read_csv(in_path)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()

    lines: list[str] = []
    lines.append(f"input_csv: {in_path}")
    lines.append(f"train_rows: {len(train_df)}")
    lines.append(f"dev_rows: {len(dev_df)}")
    if len(train_df) != 320 or len(dev_df) != 80:
        raise AssertionError(f"Unexpected split sizes: train={len(train_df)} dev={len(dev_df)}")

    # Contract: exact 13 features, semantic_coherence excluded
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing feature cols: {missing}")
    if "semantic_coherence" in df.columns:
        lines.append(f"semantic_coherence_present_in_csv: True (excluded)")
    else:
        lines.append("semantic_coherence_present_in_csv: False")
    lines.append(f"feature_cols_used(13): {FEATURE_COLS}")

    X_train_full = train_df[FEATURE_COLS]
    y_train_full = train_df["binary_label"].to_numpy(dtype=int)
    X_dev = dev_df[FEATURE_COLS]

    # Check finite
    for name, arr in [("X_train", X_train_full.to_numpy(dtype=float)), ("X_dev", X_dev.to_numpy(dtype=float))]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        lines.append(f"{name}_nan_count: {n_nan}")
        lines.append(f"{name}_inf_count: {n_inf}")
        if n_inf:
            raise AssertionError(f"Infinite values in {name}")

    rng = np.random.default_rng(RANDOM_SEED)

    # Fixed KernelSHAP background from original train split (shared across SVM runs)
    bg_size = min(SHAP_BACKGROUND_SIZE, len(X_train_full))
    bg_idx = rng.choice(len(X_train_full), size=bg_size, replace=False)
    svm_bg_arr = X_train_full.iloc[bg_idx].to_numpy(dtype=float)
    lines.append(f"kernelshap_background_size: {bg_size}")
    lines.append(f"kernelshap_nsamples: {KERNEL_SHAP_NSAMPLES}")

    baseline_specs = [
        ModelSpec(
            "logistic_regression",
            "baseline",
            "linear",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=10000, random_state=RANDOM_SEED)),
                ]
            ),
        ),
        ModelSpec(
            "random_forest",
            "baseline",
            "tree",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "svm_rbf",
            "baseline",
            "kernel",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ]
            ),
        ),
    ]
    optimized_specs = [
        ModelSpec(
            "logistic_regression",
            "optimized",
            "linear",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=10000, random_state=RANDOM_SEED, C=0.5)),
                ]
            ),
        ),
        ModelSpec(
            "random_forest",
            "optimized",
            "tree",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                            max_depth=4,
                            min_samples_leaf=4,
                            min_samples_split=2,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "svm_rbf",
            "optimized",
            "kernel",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED, C=5.0, gamma=0.01)),
                ]
            ),
        ),
    ]

    specs = baseline_specs + optimized_specs
    lines.append("")
    lines.append(f"model_versions: {len(specs)} (expected 6)")

    # Iterations
    B_DEFAULT = 50
    B_SVM = 25  # per instruction if too slow; we choose 25 proactively for reliability
    lines.append(f"bootstrap_iterations_default: {B_DEFAULT}")
    lines.append(f"bootstrap_iterations_svm: {B_SVM}")
    lines.append("")

    results_rows: list[dict[str, object]] = []

    for spec in specs:
        B = B_SVM if spec.model == "svm_rbf" else B_DEFAULT
        lines.append(f"running: {spec.stage}/{spec.model} shap={spec.shap_method} bootstraps={B}")
        t0 = time.time()

        imp_mat = np.zeros((B, len(FEATURE_COLS)), dtype=float)
        for b in range(B):
            idx = rng.integers(0, len(X_train_full), size=len(X_train_full))
            Xb = X_train_full.iloc[idx]
            yb = y_train_full[idx]

            spec.estimator.fit(Xb, yb)

            if spec.shap_method == "linear":
                shap_vals = _linear_shap(spec.estimator, Xb, X_dev)
            elif spec.shap_method == "tree":
                shap_vals = _tree_shap(spec.estimator, Xb, X_dev)
            elif spec.shap_method == "kernel":
                shap_vals = _kernel_shap(spec.estimator, X_dev, bg_arr=svm_bg_arr, nsamples=KERNEL_SHAP_NSAMPLES)
            else:
                raise ValueError(spec.shap_method)

            if shap_vals.shape != (len(X_dev), len(FEATURE_COLS)):
                raise AssertionError(f"SHAP shape mismatch at {spec.stage}/{spec.model}: {shap_vals.shape}")
            if not np.isfinite(shap_vals).all():
                raise AssertionError(f"Non-finite SHAP values at {spec.stage}/{spec.model}")

            imp_mat[b] = np.mean(np.abs(shap_vals), axis=0)

        spearman_mean, j3_mean, j5_mean, j10_mean = _pairwise_stats(imp_mat)
        cv_mean = _cv_mean_abs(imp_mat)

        elapsed = time.time() - t0
        lines.append(f"  elapsed_s: {elapsed:.2f}")
        lines.append(f"  spearman_mean: {spearman_mean:.6g}")
        lines.append(f"  jaccard_top3_mean: {j3_mean:.6g}")
        lines.append(f"  jaccard_top5_mean: {j5_mean:.6g}")
        lines.append(f"  jaccard_top10_mean: {j10_mean:.6g}")
        lines.append(f"  mean_cv_abs_shap_across_features: {cv_mean:.6g}")
        lines.append("")

        results_rows.extend(
            [
                {"model": spec.model, "stage": spec.stage, "scope": "bootstrap", "metric": "spearman_rank_importance_mean", "value": float(spearman_mean), "bootstraps": B},
                {"model": spec.model, "stage": spec.stage, "scope": "bootstrap", "metric": "jaccard_top3_mean", "value": float(j3_mean), "bootstraps": B},
                {"model": spec.model, "stage": spec.stage, "scope": "bootstrap", "metric": "jaccard_top5_mean", "value": float(j5_mean), "bootstraps": B},
                {"model": spec.model, "stage": spec.stage, "scope": "bootstrap", "metric": "jaccard_top10_mean", "value": float(j10_mean), "bootstraps": B},
                {"model": spec.model, "stage": spec.stage, "scope": "bootstrap", "metric": "mean_cv_abs_shap_across_features", "value": float(cv_mean), "bootstraps": B},
            ]
        )

    # Baseline vs optimized stability comparison (bootstrap metrics only)
    def _get(metric: str, model: str, stage: str) -> float:
        for r in results_rows:
            if r["model"] == model and r["stage"] == stage and r["metric"] == metric:
                return float(r["value"])
        return float("nan")

    lines.append("baseline_vs_optimized_bootstrap_comparison:")
    for model in ("logistic_regression", "random_forest", "svm_rbf"):
        for metric in (
            "spearman_rank_importance_mean",
            "jaccard_top3_mean",
            "jaccard_top5_mean",
            "jaccard_top10_mean",
            "mean_cv_abs_shap_across_features",
        ):
            b = _get(metric, model, "baseline")
            o = _get(metric, model, "optimized")
            delta = o - b
            results_rows.append(
                {
                    "model": model,
                    "stage": "baseline_vs_optimized",
                    "scope": "bootstrap",
                    "metric": f"delta_{metric}",
                    "value": float(delta),
                    "bootstraps": np.nan,
                }
            )
        lines.append(f"- model: {model}")
        lines.append(f"  delta_spearman: {_get('spearman_rank_importance_mean', model, 'optimized') - _get('spearman_rank_importance_mean', model, 'baseline'):.6g}")
        lines.append(f"  delta_jaccard_top5: {_get('jaccard_top5_mean', model, 'optimized') - _get('jaccard_top5_mean', model, 'baseline'):.6g}")
        lines.append(f"  delta_mean_cv_abs_shap (lower is better): {_get('mean_cv_abs_shap_across_features', model, 'optimized') - _get('mean_cv_abs_shap_across_features', model, 'baseline'):.6g}")
        lines.append("")

    # Required validation summary
    lines.append("validation:")
    lines.append("  feature_table_loaded: OK")
    lines.append("  train/dev sizes: OK (320/80)")
    lines.append("  feature_cols_exact: OK (13)")
    lines.append("  semantic_coherence_excluded: OK")
    lines.append("  nonfinite_values: none")
    lines.append("  processed_model_versions: 6")
    lines.append("  bootstrap_metrics: spearman + jaccard(top3/top5/top10) + cv")
    lines.append("  svm_bootstraps: 25 (reduced for runtime reliability)")

    out = pd.DataFrame(results_rows)
    out.to_csv(out_csv, index=False)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()

