"""Run PROCESS-2 baseline modeling only (no optimization/SHAP/stability).

Inputs:
  outputs/process2_features.csv

Outputs:
  outputs/process2_baseline_results.csv
  outputs/process2_baseline_validation_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, RANDOM_SEED


EXCLUDE_COLS = {
    "participant_id",
    "diagnosis",
    "diagnosis_label",
    "binary_label",
    "split",
    "transcript",
    "cleaned_transcript",
    "transcript_path",
    "audio_path",
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path = OUTPUT_DIR / "process2_features.csv"
    out_csv = OUTPUT_DIR / "process2_baseline_results.csv"
    out_report = OUTPUT_DIR / "process2_baseline_validation_report.txt"

    df = pd.read_csv(in_path)

    lines: list[str] = []
    lines.append(f"input_csv: {in_path}")
    lines.append(f"rows_total: {len(df)}")

    # Basic split checks
    if "split" not in df.columns:
        raise AssertionError("Missing split column in process2_features.csv")
    if "binary_label" not in df.columns:
        raise AssertionError("Missing binary_label column in process2_features.csv")
    if "participant_id" not in df.columns:
        raise AssertionError("Missing participant_id column in process2_features.csv")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()

    lines.append(f"train_rows: {len(train_df)}")
    lines.append(f"dev_rows: {len(dev_df)}")
    if len(train_df) != 320:
        raise AssertionError(f"Expected train=320, got {len(train_df)}")
    if len(dev_df) != 80:
        raise AssertionError(f"Expected dev=80, got {len(dev_df)}")

    # Label distributions
    lines.append(f"train_binary_counts: {train_df['binary_label'].value_counts().sort_index().to_dict()}")
    lines.append(f"dev_binary_counts: {dev_df['binary_label'].value_counts().sort_index().to_dict()}")

    # No participant overlap
    train_ids = set(train_df["participant_id"].astype(str))
    dev_ids = set(dev_df["participant_id"].astype(str))
    overlap = sorted(train_ids & dev_ids)
    lines.append(f"participant_overlap_count: {len(overlap)}")
    if overlap:
        raise AssertionError(f"Participant IDs overlap between train and dev: {overlap[:5]}")

    # Feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if "semantic_coherence" in feature_cols:
        # Exclude if constant 0.0 (current environment fallback)
        sc = df["semantic_coherence"]
        is_const_zero = bool((sc == 0.0).all())
        lines.append(f"semantic_coherence_constant_zero: {is_const_zero}")
        feature_cols = [c for c in feature_cols if c != "semantic_coherence"]
    else:
        lines.append("semantic_coherence_present: False")

    lines.append(f"n_feature_cols: {len(feature_cols)}")
    lines.append(f"feature_cols: {feature_cols}")

    # Ensure all features numeric
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    lines.append(f"non_numeric_feature_cols: {non_numeric}")
    if non_numeric:
        raise AssertionError(f"Non-numeric feature columns in X: {non_numeric}")

    # Ensure no NaN/inf in X
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_dev = dev_df[feature_cols].to_numpy(dtype=float)
    for name, arr in [("X_train", X_train), ("X_dev", X_dev)]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        lines.append(f"{name}_nan_count: {n_nan}")
        lines.append(f"{name}_inf_count: {n_inf}")
        if n_inf:
            raise AssertionError(f"Infinite values in {name}")

    y_train = train_df["binary_label"].to_numpy(dtype=int)
    y_dev = dev_df["binary_label"].to_numpy(dtype=int)

    models: list[tuple[str, Pipeline]] = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=10000, random_state=RANDOM_SEED)),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
                        ),
                    ),
                ]
            ),
        ),
        (
            "svm_rbf",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ]
            ),
        ),
    ]

    rows: list[dict[str, object]] = []
    for model_name, pipe in models:
        pipe.fit(train_df[feature_cols], y_train)
        y_pred = pipe.predict(dev_df[feature_cols])
        y_score = pipe.predict_proba(dev_df[feature_cols])[:, 1]
        m = _metrics(y_dev, y_pred, y_score)
        tn, fp, fn, tp = confusion_matrix(y_dev, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": model_name,
                "split": "dev",
                **m,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        lines.append("")
        lines.append(f"model: {model_name}")
        for k, v in m.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"  confusion_matrix(TN,FP,FN,TP): {tn},{fp},{fn},{tp}")

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    lines.insert(1, f"output_csv: {out_csv}")
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()

