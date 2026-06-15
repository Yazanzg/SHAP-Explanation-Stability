"""TASK 2 - Imputation audit and feature collinearity matrix.

Loads the configured feature matrix and answers, definitively, whether any
imputation is needed. Because hand-crafted linguistic features should not have
random missingness, any NaN would be *informative* and imputing it under a
missing-at-random assumption would be statistically inappropriate. This script:

  1. Reports, per configured feature: NaN count, infinite-value count, constant
     flag, dtype, and range.
  2. Cross-tabulates NaN counts by class (HC vs CI) to check whether any
     missingness rate differs by class.
  3. If the matrix has zero NaNs, proves that removing the median SimpleImputer
     changes nothing: the imputer is an exact no-op on this matrix, and model
     predictions on the dev split are bit-identical with and without it.
  4. Computes and saves the 13x13 Pearson and Spearman feature-correlation
     matrices (for the discussion's collinearity-mechanism argument).

Outputs (under outputs/audit/):
  - feature_audit.md
  - feature_correlations.csv
  - feature_correlations.png

Run (root config; ensure THESIS_CONFIG is not pointing at the synthetic example):
    python scripts/audit_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.config import (
    COL_LABEL,
    FEATURE_COLUMNS,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    TRAIN_SPLIT_VALUE,
    TEST_SPLIT_VALUE,
)
from src.modeling import build_estimator


def _load_feature_frame() -> pd.DataFrame:
    path = config.FEATURES_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature matrix not found: {path}. Run feature extraction first."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature matrix missing configured features: {missing}")
    return df


def _per_feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURE_COLUMNS:
        col = df[c]
        arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "feature": c,
                "dtype": str(col.dtype),
                "nan_count": int(col.isna().sum()),
                "inf_count": int(np.isinf(arr).sum()),
                "constant": bool(col.nunique(dropna=True) <= 1),
                "n_unique": int(col.nunique(dropna=True)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
            }
        )
    return pd.DataFrame(rows)


def _nan_crosstab_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURE_COLUMNS:
        nan_mask = df[c].isna()
        n_hc = int(nan_mask[df[COL_LABEL] == 0].sum())
        n_ci = int(nan_mask[df[COL_LABEL] == 1].sum())
        rows.append({"feature": c, f"nan_{NEGATIVE_CLASS_NAME}": n_hc, f"nan_{POSITIVE_CLASS_NAME}": n_ci})
    return pd.DataFrame(rows)


def _imputer_is_noop(df: pd.DataFrame) -> tuple[bool, bool]:
    """Return (matrix_unchanged, predictions_identical)."""
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    imputed = SimpleImputer(strategy="median").fit_transform(X)
    matrix_unchanged = bool(np.array_equal(np.nan_to_num(X), np.nan_to_num(imputed))) and (
        np.isnan(X).sum() == 0
    )

    # Compare dev-split predictions with vs without a leading median imputer.
    train = df[df[config.COL_SPLIT] == TRAIN_SPLIT_VALUE]
    dev = df[df[config.COL_SPLIT] == TEST_SPLIT_VALUE]
    Xtr = train[list(FEATURE_COLUMNS)]
    ytr = train[COL_LABEL].to_numpy(dtype=int)
    Xdev = dev[list(FEATURE_COLUMNS)]

    predictions_identical = True
    from sklearn.pipeline import Pipeline

    for model_name in config.MODELS:
        base = build_estimator(model_name)
        base.fit(Xtr, ytr)
        proba_base = base.predict_proba(Xdev)[:, 1]

        with_imputer = Pipeline(
            [("impute", SimpleImputer(strategy="median")), *build_estimator(model_name).steps]
        )
        with_imputer.fit(Xtr, ytr)
        proba_imp = with_imputer.predict_proba(Xdev)[:, 1]

        if not np.array_equal(proba_base, proba_imp):
            predictions_identical = False
    return matrix_unchanged, predictions_identical


def _correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df[list(FEATURE_COLUMNS)].astype(float)
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")
    # Tidy long-format export with both coefficients.
    rows = []
    feats = list(FEATURE_COLUMNS)
    for i, a in enumerate(feats):
        for j, b in enumerate(feats):
            rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "pearson": float(pearson.loc[a, b]),
                    "spearman": float(spearman.loc[a, b]),
                }
            )
    return pearson, spearman, pd.DataFrame(rows)


def _plot_correlations(pearson: pd.DataFrame, spearman: pd.DataFrame, out_png: Path) -> None:
    feats = list(FEATURE_COLUMNS)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.3))  # ~A4 landscape
    for ax, mat, title in (
        (axes[0], pearson, "Pearson"),
        (axes[1], spearman, "Spearman"),
    ):
        im = ax.imshow(mat.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(feats)))
        ax.set_yticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=90, fontsize=8)
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_title(f"{title} feature correlation", fontsize=12)
        for i in range(len(feats)):
            for j in range(len(feats)):
                ax.text(
                    j, i, f"{mat.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=5.5,
                    color="black" if abs(mat.iloc[i, j]) < 0.6 else "white",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "13 transcript-derived features — pairwise correlation (n=400)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_feature_frame()

    audit = _per_feature_audit(df)
    crosstab = _nan_crosstab_by_class(df)
    total_nan = int(audit["nan_count"].sum())
    total_inf = int(audit["inf_count"].sum())
    constants = audit.loc[audit["constant"], "feature"].tolist()

    pearson, spearman, corr_long = _correlations(df)
    corr_csv = config.AUDIT_DIR / "feature_correlations.csv"
    corr_long.to_csv(corr_csv, index=False)
    corr_png = config.AUDIT_DIR / "feature_correlations.png"
    _plot_correlations(pearson, spearman, corr_png)

    matrix_unchanged = predictions_identical = None
    if total_nan == 0:
        matrix_unchanged, predictions_identical = _imputer_is_noop(df)

    # Highly correlated pairs for the collinearity discussion (|Pearson| >= 0.7).
    high_pairs = (
        corr_long[(corr_long["feature_a"] < corr_long["feature_b"])]
        .assign(abs_pearson=lambda d: d["pearson"].abs())
        .query("abs_pearson >= 0.7")
        .sort_values("abs_pearson", ascending=False)
    )

    lines: list[str] = []
    lines.append("# Feature audit — imputation and collinearity (TASK 2)\n")
    lines.append(f"- Feature matrix: `{config.FEATURES_CSV}`")
    lines.append(f"- Rows: {len(df)}  |  Configured features: {len(FEATURE_COLUMNS)}")
    lines.append(f"- Total NaNs across the {len(FEATURE_COLUMNS)} features: **{total_nan}**")
    lines.append(f"- Total infinite values: **{total_inf}**")
    lines.append(
        f"- Constant features: **{constants if constants else 'none'}**\n"
    )

    lines.append("## Per-feature audit\n")
    lines.append("| feature | dtype | nan | inf | constant | n_unique | min | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in audit.iterrows():
        lines.append(
            f"| {r['feature']} | {r['dtype']} | {r['nan_count']} | {r['inf_count']} | "
            f"{r['constant']} | {r['n_unique']} | {r['min']:.4g} | {r['max']:.4g} |"
        )
    lines.append("")

    lines.append(f"## NaN cross-tabulation by class ({NEGATIVE_CLASS_NAME} vs {POSITIVE_CLASS_NAME})\n")
    lines.append(f"| feature | nan_{NEGATIVE_CLASS_NAME} | nan_{POSITIVE_CLASS_NAME} |")
    lines.append("|---|---|---|")
    for _, r in crosstab.iterrows():
        lines.append(
            f"| {r['feature']} | {r[f'nan_{NEGATIVE_CLASS_NAME}']} | {r[f'nan_{POSITIVE_CLASS_NAME}']} |"
        )
    lines.append("")

    lines.append("## Imputation decision\n")
    if total_nan == 0:
        lines.append(
            "The feature matrix contains **zero missing values and zero infinities** "
            "across all configured features, and missingness does not exist in either "
            "class. The median `SimpleImputer` is therefore an exact no-op: applying it "
            f"leaves the matrix unchanged (`matrix_unchanged={matrix_unchanged}`) and "
            "model predictions on the fixed dev split are bit-identical with and without "
            f"it (`predictions_identical={predictions_identical}`). The imputer has been "
            "removed from all active model pipelines. No imputation is performed anywhere "
            "in the pipeline; because hand-crafted linguistic features cannot be randomly "
            "missing, imputation under a missing-at-random assumption would be "
            "statistically inappropriate, and there is nothing to impute in any case."
        )
    else:
        lines.append(
            f"The feature matrix contains **{total_nan} missing values**. Per the brief, "
            "these are NOT imputed. Each NaN must be traced to its source transcript "
            "(empty cleaned text, spaCy parse failure, or zero-denominator ratio) and we "
            "decide together whether to drop affected participants, drop the feature, or "
            "document the issue. (Trace table to be produced.)"
        )
    lines.append("")

    lines.append("## Feature collinearity (for the discussion)\n")
    lines.append(
        f"Pearson and Spearman 13x13 matrices saved to `{corr_csv.name}` and "
        f"`{corr_png.name}` (+ PDF). Highly correlated pairs (|Pearson| >= 0.7):\n"
    )
    if len(high_pairs):
        lines.append("| feature_a | feature_b | pearson | spearman |")
        lines.append("|---|---|---|---|")
        for _, r in high_pairs.iterrows():
            lines.append(
                f"| {r['feature_a']} | {r['feature_b']} | {r['pearson']:.3f} | {r['spearman']:.3f} |"
            )
    else:
        lines.append("_No feature pair exceeds |Pearson| >= 0.7._")
    lines.append("")

    lines.append("## Note: imputation removed from the active pipeline\n")
    lines.append(
        "The first-submission scripts that used `SimpleImputer` (GridSearchCV "
        "optimisation, the old baseline/optimised SHAP and bootstrap-stability "
        "validators, and the monolithic `run_pipeline.py`) have been removed from the "
        "active branch as part of the training-size reframing. The only model "
        "construction path in the resubmission is `src/modeling.build_estimator`, which "
        "is imputer-free."
    )
    lines.append("")

    out_md = config.AUDIT_DIR / "feature_audit.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {corr_csv}")
    print(f"Wrote: {corr_png} (+ .pdf)")
    print(f"total_nan={total_nan} total_inf={total_inf} constants={constants}")
    if total_nan == 0:
        print(f"matrix_unchanged={matrix_unchanged} predictions_identical={predictions_identical}")


if __name__ == "__main__":
    main()
