"""Explanation stability analysis for PROCESS-2 using existing SHAP CSVs only.

Inputs (must already exist under outputs/shap/):
  - process2_shap_values_{baseline|optimized}_{logistic_regression|random_forest|svm_rbf}.csv

Outputs:
  - outputs/process2_stability_results.csv
  - outputs/process2_stability_validation_report.txt

This script does NOT retrain models and does NOT regenerate SHAP values.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR


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
class ShapFile:
    stage: str  # baseline/optimized
    model: str  # logistic_regression/random_forest/svm_rbf
    path: Path


FILES: list[ShapFile] = [
    ShapFile("baseline", "logistic_regression", OUTPUT_DIR / "shap" / "process2_shap_values_baseline_logistic_regression.csv"),
    ShapFile("optimized", "logistic_regression", OUTPUT_DIR / "shap" / "process2_shap_values_optimized_logistic_regression.csv"),
    ShapFile("baseline", "random_forest", OUTPUT_DIR / "shap" / "process2_shap_values_baseline_random_forest.csv"),
    ShapFile("optimized", "random_forest", OUTPUT_DIR / "shap" / "process2_shap_values_optimized_random_forest.csv"),
    ShapFile("baseline", "svm_rbf", OUTPUT_DIR / "shap" / "process2_shap_values_baseline_svm_rbf.csv"),
    ShapFile("optimized", "svm_rbf", OUTPUT_DIR / "shap" / "process2_shap_values_optimized_svm_rbf.csv"),
]


def _jaccard_topk(a: pd.Series, b: pd.Series, k: int) -> float:
    ta = set(a.nlargest(k).index)
    tb = set(b.nlargest(k).index)
    if not ta and not tb:
        return 1.0
    return float(len(ta & tb) / max(1, len(ta | tb)))


def _mean_abs_importance(df: pd.DataFrame) -> pd.Series:
    return df[FEATURE_COLS].abs().mean(axis=0)


def _cv_abs_per_feature(df: pd.DataFrame) -> pd.Series:
    """Coefficient of variation of |SHAP| across dev participants (per feature)."""
    abs_vals = df[FEATURE_COLS].abs()
    mu = abs_vals.mean(axis=0)
    sd = abs_vals.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = sd / mu
    cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cv


def _rank_series(s: pd.Series) -> pd.Series:
    """Rank features (1=most important) based on descending s."""
    return s.rank(ascending=False, method="average")


def main() -> None:
    out_csv = OUTPUT_DIR / "process2_stability_results.csv"
    out_report = OUTPUT_DIR / "process2_stability_validation_report.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    rows: list[dict[str, object]] = []

    lines.append("PROCESS-2 explanation stability (baseline vs optimized) using dev SHAP only")
    lines.append(f"feature_cols(13): {FEATURE_COLS}")
    lines.append("")

    loaded: dict[tuple[str, str], pd.DataFrame] = {}

    # -----------------------
    # Load + validate each file
    # -----------------------
    for sf in FILES:
        if not sf.path.is_file():
            raise FileNotFoundError(f"Missing SHAP file: {sf.path}")
        df = pd.read_csv(sf.path)
        loaded[(sf.stage, sf.model)] = df

        lines.append(f"loaded: {sf.path.name} rows={len(df)} cols={len(df.columns)}")
        if len(df) != 80:
            raise AssertionError(f"{sf.path.name} expected 80 rows (dev), got {len(df)}")
        for c in ("participant_id", "split", "binary_label"):
            if c not in df.columns:
                raise AssertionError(f"{sf.path.name} missing required column: {c}")
        if "semantic_coherence" in df.columns:
            raise AssertionError(f"{sf.path.name} must not include semantic_coherence")
        if list([c for c in FEATURE_COLS if c in df.columns]) != FEATURE_COLS:
            raise AssertionError(f"{sf.path.name} feature columns mismatch")
        arr = df[FEATURE_COLS].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise AssertionError(f"{sf.path.name} contains non-finite SHAP values")

        # Stash per-model-version summaries
        mean_abs = _mean_abs_importance(df)
        cv_abs = _cv_abs_per_feature(df)
        rows.append(
            {
                "model": sf.model,
                "stage": sf.stage,
                "metric": "mean_cv_abs_shap_across_features",
                "value": float(cv_abs.mean()),
            }
        )
        # Top features (for report)
        top10 = mean_abs.sort_values(ascending=False).head(10)
        lines.append(f"  expected_value_col_present: {'expected_value' in df.columns}")
        lines.append("  top10_mean_abs_shap:")
        for feat, val in top10.items():
            lines.append(f"    {feat}: {float(val):.6g}")
        lines.append("")

    # -----------------------
    # Pairwise stability: baseline vs optimized per model
    # -----------------------
    for model in ("logistic_regression", "random_forest", "svm_rbf"):
        b = loaded[("baseline", model)].copy()
        o = loaded[("optimized", model)].copy()

        # Participant alignment check
        b_ids = set(b["participant_id"].astype(str))
        o_ids = set(o["participant_id"].astype(str))
        if b_ids != o_ids:
            raise AssertionError(f"participant_id sets differ for {model}: baseline vs optimized")

        # Align row order for any potential per-row comparisons later
        b = b.sort_values("participant_id").reset_index(drop=True)
        o = o.sort_values("participant_id").reset_index(drop=True)
        if not (b["participant_id"].astype(str).values == o["participant_id"].astype(str).values).all():
            raise AssertionError(f"participant_id alignment failed after sorting for {model}")

        mean_abs_b = _mean_abs_importance(b)
        mean_abs_o = _mean_abs_importance(o)

        # Spearman on feature ranking vectors (importance ordering)
        rb = _rank_series(mean_abs_b)
        ro = _rank_series(mean_abs_o)
        sp, _ = spearmanr(rb.values, ro.values)
        sp_val = float(sp) if not np.isnan(sp) else 0.0
        rows.append(
            {
                "model": model,
                "stage": "baseline_vs_optimized",
                "metric": "spearman_rank_importance",
                "value": sp_val,
            }
        )

        # Jaccard top-k on mean_abs sets
        for k in (3, 5, 10):
            jac = _jaccard_topk(mean_abs_b, mean_abs_o, k=k)
            rows.append(
                {
                    "model": model,
                    "stage": "baseline_vs_optimized",
                    "metric": f"jaccard_top{k}",
                    "value": float(jac),
                }
            )

        # CV comparison (mean CV across features)
        cv_b = float(_cv_abs_per_feature(b).mean())
        cv_o = float(_cv_abs_per_feature(o).mean())
        rows.append(
            {
                "model": model,
                "stage": "baseline_vs_optimized",
                "metric": "delta_mean_cv_abs_shap_across_features",
                "value": float(cv_o - cv_b),
            }
        )

        # Summary direction (not a thesis conclusion; just sign)
        def _dir(x: float, eps: float = 1e-6) -> str:
            if x > eps:
                return "increased"
            if x < -eps:
                return "decreased"
            return "preserved"

        # Higher spearman/jaccard => more stable; lower CV => more stable
        cv_dir = _dir(-(cv_o - cv_b))  # invert: lower CV is better
        lines.append(f"model: {model}")
        lines.append(f"  spearman_rank_importance: {sp_val:.6g}")
        lines.append(f"  jaccard_top3/top5/top10: "
                     f"{_jaccard_topk(mean_abs_b, mean_abs_o, 3):.6g} / "
                     f"{_jaccard_topk(mean_abs_b, mean_abs_o, 5):.6g} / "
                     f"{_jaccard_topk(mean_abs_b, mean_abs_o, 10):.6g}")
        lines.append(f"  mean_cv_abs_shap baseline: {cv_b:.6g}")
        lines.append(f"  mean_cv_abs_shap optimized: {cv_o:.6g}")
        lines.append(f"  mean_cv_abs_shap stability_direction: {cv_dir}")
        lines.append("")

    # Final validation summary
    lines.append("validation:")
    lines.append("  loaded_files: 6 (expected 6)")
    lines.append("  rows_per_file: 80 (checked)")
    lines.append("  feature_columns_exact_match: OK")
    lines.append("  nonfinite_shap_values: none")
    lines.append("  computed_metrics: spearman + jaccard(top3/top5/top10) + cv")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()

