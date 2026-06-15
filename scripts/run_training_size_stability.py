"""Training-size learning-curve runner: per-subset performance + SHAP vectors.

For each model and each training size, draw `iterations` balanced, stratified
subsets WITHOUT replacement from the training pool (n_hc = n_ci = size/2), fit a
fixed-setting model, evaluate on the FIXED dev split, and reduce each run's SHAP
to a mean-|SHAP| importance vector of length 13. The dev split is never resampled.

Two modes:
  --smoke  use experiment.smoke (TASK 4 quick check: LR, 3 sizes, 5 repeats)
  (default) use the full experiment grid (TASK 6)

Outputs (under outputs/, tagged smoke/full):
  <tag>_performance.csv                          dev metrics per (model, size, run)
  importance_vectors/<tag>_importance_vectors.csv  mean-|SHAP| per (model, size, run, feature)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.config import (
    COL_LABEL,
    COL_PARTICIPANT_ID,
    COL_SPLIT,
    FEATURE_COLUMNS,
    TRAIN_SPLIT_VALUE,
    TEST_SPLIT_VALUE,
    iteration_seed,
)
from src.explainability import compute_shap_values
from src.modeling import build_estimator, evaluate_on_split


def _balanced_subset(pool: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Draw size/2 HC (label 0) + size/2 CI (label 1) without replacement.

    The concatenated subset is then placed in a deterministic order (by
    participant_id when available, otherwise the original master-table order).
    This guarantees attribution differences between runs reflect *which*
    participants were sampled, not incidental row order -- which matters because
    some estimators (e.g. Random Forest bootstrap) are row-order dependent.
    """
    n = size // 2
    hc = pool[pool[COL_LABEL] == 0]
    ci = pool[pool[COL_LABEL] == 1]
    if len(hc) < n or len(ci) < n:
        raise ValueError(f"size={size} needs {n}/class; pool has {len(hc)} HC / {len(ci)} CI")
    sub_hc = hc.sample(n=n, random_state=seed, replace=False)
    sub_ci = ci.sample(n=n, random_state=seed, replace=False)
    sub = pd.concat([sub_hc, sub_ci])
    if COL_PARTICIPANT_ID in sub.columns:
        sub = sub.sort_values(COL_PARTICIPANT_ID, kind="mergesort")
    else:
        sub = sub.sort_index(kind="mergesort")
    return sub.reset_index(drop=True)


def _importance_vector(shap_long: pd.DataFrame) -> dict[str, float]:
    g = shap_long.groupby("feature_name")["shap_value"].apply(lambda s: s.abs().mean())
    return {f: float(g.get(f, np.nan)) for f in FEATURE_COLUMNS}


def _pairwise_pearson(vectors: list[np.ndarray]) -> list[float]:
    out: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="use experiment.smoke settings")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these model keys (e.g. logistic_regression random_forest)")
    args = ap.parse_args()

    if args.smoke:
        tag = "smoke"
        sizes = config.SMOKE_TRAINING_SIZE_GRID
        iters = config.SMOKE_ITERATIONS
        models = args.models or config.SMOKE_MODELS
    else:
        tag = "full"
        sizes = config.TRAINING_SIZE_GRID
        iters = config.ITERATIONS_PER_SIZE
        models = args.models or list(config.MODELS)
    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"Unknown model '{m}'. Configured: {list(config.MODELS)}")

    df = pd.read_csv(config.FEATURES_CSV)
    df[COL_SPLIT] = df[COL_SPLIT].astype(str).str.strip().str.lower()
    pool = df[df[COL_SPLIT] == TRAIN_SPLIT_VALUE].reset_index(drop=True)
    # Fixed dev/test set in deterministic order (by participant_id when available)
    # so the held-out evaluation and the SHAP background are stable across runs.
    dev = df[df[COL_SPLIT] == TEST_SPLIT_VALUE]
    if COL_PARTICIPANT_ID in dev.columns:
        dev = dev.sort_values(COL_PARTICIPANT_ID, kind="mergesort")
    dev = dev.reset_index(drop=True)
    dev[COL_PARTICIPANT_ID] = dev[COL_PARTICIPANT_ID].astype(str)

    print(f"[run:{tag}] models={models} sizes={sizes} iters={iters}")
    print(f"[run:{tag}] pool={len(pool)} (HC={int((pool[COL_LABEL]==0).sum())}, "
          f"CI={int((pool[COL_LABEL]==1).sum())})  dev={len(dev)}")

    perf_rows: list[dict] = []
    imp_rows: list[dict] = []
    t0 = time.time()

    for model in models:
        for size_index, size in enumerate(sizes):
            for run_id in range(iters):
                seed = iteration_seed(size_index, run_id)
                sub = _balanced_subset(pool, size, seed)
                pipe = build_estimator(model)
                pipe.fit(sub[list(FEATURE_COLUMNS)], sub[COL_LABEL].to_numpy(int))
                perf = evaluate_on_split(pipe, dev[list(FEATURE_COLUMNS)], dev[COL_LABEL].to_numpy(int))
                perf_rows.append({"model": model, "training_size": size, "run_id": run_id,
                                  "seed": seed, **perf})
                shap_long = compute_shap_values(
                    model, pipe, dev, variant=f"{model}_size{size}_run{run_id}"
                )
                vec = _importance_vector(shap_long)
                for f, v in vec.items():
                    imp_rows.append({"model": model, "training_size": size, "run_id": run_id,
                                     "seed": seed, "feature_name": f, "mean_abs_shap": v})

    perf_df = pd.DataFrame(perf_rows)
    imp_df = pd.DataFrame(imp_rows)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.IMPORTANCE_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    if tag == "full":
        perf_path = config.OUTPUT_DIR / "training_size_runs.csv"
        imp_path = config.IMPORTANCE_VECTORS_DIR / "training_size_importance_vectors.csv"
    else:
        perf_path = config.OUTPUT_DIR / f"{tag}_performance.csv"
        imp_path = config.IMPORTANCE_VECTORS_DIR / f"{tag}_importance_vectors.csv"
    perf_df.to_csv(perf_path, index=False)
    imp_df.to_csv(imp_path, index=False)

    lo, hi = config.PERCENTILE_INTERVAL
    n_vectors = imp_df[["model", "training_size", "run_id"]].drop_duplicates().shape[0]
    print(f"\n[run:{tag}] completed {len(perf_df)} fits in {time.time()-t0:.1f}s")
    print(f"[run:{tag}] runs rows={len(perf_df)}  importance vectors={n_vectors}  "
          f"importance rows={len(imp_df)}")
    print(f"[run:{tag}] wrote {perf_path}")
    print(f"[run:{tag}] wrote {imp_path}\n")

    print("=== Dev performance (mean over runs) ===")
    pm = perf_df.groupby(["model", "training_size"]).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        f1_mean=("f1", "mean"), acc_mean=("accuracy", "mean"),
    ).reset_index()
    print(pm.to_string(index=False))

    print("\n=== Attribution-stability preview (pairwise Pearson of mean-|SHAP| vectors) ===")
    for model in models:
        for size in sizes:
            cell = imp_df[(imp_df.model == model) & (imp_df.training_size == size)]
            vecs = [
                cell[cell.run_id == r].set_index("feature_name")
                    .reindex(FEATURE_COLUMNS)["mean_abs_shap"].to_numpy(float)
                for r in sorted(cell.run_id.unique())
            ]
            pp = _pairwise_pearson(vecs)
            if pp:
                arr = np.array(pp)
                print(f"  {model:<20} size={size:>3}  pearson mean={arr.mean():.4f}  "
                      f"[{np.percentile(arr, lo):.4f}, {np.percentile(arr, hi):.4f}]  "
                      f"(n_pairs={len(pp)})")
            else:
                print(f"  {model:<20} size={size:>3}  pearson n/a (degenerate vectors)")


if __name__ == "__main__":
    main()
