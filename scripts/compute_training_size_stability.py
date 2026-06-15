"""TASK 5: summarise attribution stability and performance vs training size.

Reads the per-run importance vectors and dev-performance produced by
scripts/run_training_size_stability.py and computes, per (model, training_size):

  * attribution stability over all C(B, 2) pairwise comparisons of the B
    mean-|SHAP| vectors -- Pearson (primary), Spearman (appendix), and Jaccard@k
    on the top-k features (k from config.metrics.jaccard_ks);
  * dev performance central tendency with a percentile band.

Both summaries report the mean and the configured percentile interval (e.g.
2.5 / 97.5 -> 95% band).

Outputs:
  outputs/training_size_stability_summary.csv
  outputs/training_size_performance_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.config import FEATURE_COLUMNS, JACCARD_KS, PERCENTILE_INTERVAL

PERF_METRICS = ["auroc", "f1", "accuracy", "precision", "recall"]


def _band(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, low_pct, high_pct) for a list of values; NaNs ignored."""
    arr = np.asarray([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = PERCENTILE_INTERVAL
    return (float(arr.mean()), float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def _jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a))
    sa = set(np.argsort(a)[::-1][:k].tolist())
    sb = set(np.argsort(b)[::-1][:k].tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


def _vectors_for_cell(cell: pd.DataFrame) -> list[np.ndarray]:
    vecs = []
    for r in sorted(cell["run_id"].unique()):
        v = (
            cell[cell["run_id"] == r]
            .set_index("feature_name")
            .reindex(FEATURE_COLUMNS)["mean_abs_shap"]
            .to_numpy(dtype=float)
        )
        vecs.append(v)
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke", choices=["smoke", "full"],
                    help="which learning-curve outputs to summarise")
    args = ap.parse_args()

    imp_path = config.IMPORTANCE_VECTORS_DIR / f"{args.tag}_importance_vectors.csv"
    perf_path = config.OUTPUT_DIR / f"{args.tag}_performance.csv"
    imp = pd.read_csv(imp_path)
    perf = pd.read_csv(perf_path)

    # ---- Stability summary ----
    stab_rows: list[dict] = []
    for (model, size), cell in imp.groupby(["model", "training_size"]):
        vecs = _vectors_for_cell(cell)
        b = len(vecs)
        pairs = list(combinations(range(b), 2))
        pear, spear = [], []
        jac = {k: [] for k in JACCARD_KS}
        for i, j in pairs:
            a, c = vecs[i], vecs[j]
            if np.std(a) > 1e-12 and np.std(c) > 1e-12:
                pear.append(float(np.corrcoef(a, c)[0, 1]))
                spear.append(float(spearmanr(a, c).statistic))
            for k in JACCARD_KS:
                jac[k].append(_jaccard_topk(a, c, k))
        p_mean, p_lo, p_hi = _band(pear)
        s_mean, s_lo, s_hi = _band(spear)
        row = {
            "model": model, "training_size": size,
            "n_repeats": b, "n_pairs": len(pairs),
            "pearson_mean": p_mean, "pearson_lo": p_lo, "pearson_hi": p_hi,
            "spearman_mean": s_mean, "spearman_lo": s_lo, "spearman_hi": s_hi,
        }
        for k in JACCARD_KS:
            jm, jl, jh = _band(jac[k])
            row[f"jaccard{k}_mean"] = jm
            row[f"jaccard{k}_lo"] = jl
            row[f"jaccard{k}_hi"] = jh
        stab_rows.append(row)
    stab_df = pd.DataFrame(stab_rows).sort_values(["model", "training_size"]).reset_index(drop=True)

    # ---- Performance summary ----
    perf_rows: list[dict] = []
    for (model, size), cell in perf.groupby(["model", "training_size"]):
        row = {"model": model, "training_size": size, "n_repeats": len(cell)}
        for m in PERF_METRICS:
            mean, lo, hi = _band(cell[m].tolist())
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        perf_rows.append(row)
    perf_df = pd.DataFrame(perf_rows).sort_values(["model", "training_size"]).reset_index(drop=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stab_out = config.OUTPUT_DIR / "training_size_stability_summary.csv"
    perf_out = config.OUTPUT_DIR / "training_size_performance_summary.csv"
    stab_df.to_csv(stab_out, index=False)
    perf_df.to_csv(perf_out, index=False)

    # ---- Pairwise-count sanity check ----
    bad = stab_df[stab_df["n_pairs"] != stab_df["n_repeats"] * (stab_df["n_repeats"] - 1) // 2]
    print(f"wrote: {stab_out}")
    print(f"wrote: {perf_out}")
    print(f"pairwise-count check: all cells have C(B,2) pairs -> {bad.empty}")
    print("\n=== training_size_stability_summary.csv ===")
    print(stab_df.to_string(index=False))
    print("\n=== training_size_performance_summary.csv ===")
    print(perf_df.to_string(index=False))


if __name__ == "__main__":
    main()
