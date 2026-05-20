"""Controlled optimization (GridSearchCV) for PROCESS-2 baselines only.

Inputs:
  outputs/process2_features.csv
  outputs/process2_baseline_results.csv (for comparison; optional)

Outputs:
  outputs/process2_optimized_results.csv
  outputs/process2_optimization_validation_report.txt

Notes:
  - Train split only for model selection.
  - Dev split evaluated once for final optimized metrics.
  - No SHAP, stability, or plots are generated here.
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, RANDOM_SEED


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _check_X(arr: np.ndarray, *, name: str, lines: list[str]) -> None:
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    lines.append(f"{name}_nan_count: {n_nan}")
    lines.append(f"{name}_inf_count: {n_inf}")
    if n_inf:
        raise AssertionError(f"Infinite values in {name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path = OUTPUT_DIR / "process2_features.csv"
    base_path = OUTPUT_DIR / "process2_baseline_results.csv"
    out_csv = OUTPUT_DIR / "process2_optimized_results.csv"
    out_report = OUTPUT_DIR / "process2_optimization_validation_report.txt"

    df = pd.read_csv(in_path)
    if "split" not in df.columns or "binary_label" not in df.columns or "participant_id" not in df.columns:
        raise AssertionError("process2_features.csv missing required columns (split/binary_label/participant_id)")

    lines: list[str] = []
    lines.append(f"input_csv: {in_path}")
    lines.append(f"baseline_csv_exists: {base_path.is_file()}")
    lines.append(f"rows_total: {len(df)}")

    # Split checks
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    lines.append(f"train_rows: {len(train_df)}")
    lines.append(f"dev_rows: {len(dev_df)}")
    if len(train_df) != 320:
        raise AssertionError(f"Expected train=320, got {len(train_df)}")
    if len(dev_df) != 80:
        raise AssertionError(f"Expected dev=80, got {len(dev_df)}")

    # No participant overlap
    overlap = set(train_df["participant_id"].astype(str)) & set(dev_df["participant_id"].astype(str))
    lines.append(f"participant_overlap_count: {len(overlap)}")
    if overlap:
        raise AssertionError(f"Participant overlap between train and dev: {list(overlap)[:5]}")

    # Confirm semantic_coherence excluded and only approved cols used
    if "semantic_coherence" in FEATURE_COLS:
        raise AssertionError("semantic_coherence should not be in FEATURE_COLS")
    missing_feat = [c for c in FEATURE_COLS if c not in df.columns]
    lines.append(f"missing_feature_cols: {missing_feat}")
    if missing_feat:
        raise AssertionError(f"Missing feature columns in CSV: {missing_feat}")

    extra_feat_like = [c for c in FEATURE_COLS if c == "semantic_coherence"]
    lines.append(f"semantic_coherence_in_feature_list: {bool(extra_feat_like)}")
    lines.append(f"feature_cols_used: {FEATURE_COLS}")

    # X/y and numeric checks
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    X_dev = dev_df[FEATURE_COLS].to_numpy(dtype=float)
    _check_X(X_train, name="X_train", lines=lines)
    _check_X(X_dev, name="X_dev", lines=lines)
    y_train = train_df["binary_label"].to_numpy(dtype=int)
    y_dev = dev_df["binary_label"].to_numpy(dtype=int)

    lines.append(f"train_binary_counts: {train_df['binary_label'].value_counts().sort_index().to_dict()}")
    lines.append(f"dev_binary_counts: {dev_df['binary_label'].value_counts().sort_index().to_dict()}")

    # Baseline comparison (optional)
    baseline_map: dict[str, dict[str, float]] = {}
    if base_path.is_file():
        base = pd.read_csv(base_path)
        for _, r in base.iterrows():
            baseline_map[str(r["model"])] = {
                "auroc": float(r["auroc"]),
                "accuracy": float(r["accuracy"]),
                "f1": float(r["f1"]),
            }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # --- Logistic Regression ---
    lr = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=10000, random_state=RANDOM_SEED)),
        ]
    )
    lr_grid = {"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    lr_gs = GridSearchCV(lr, lr_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    lr_gs.fit(train_df[FEATURE_COLS], y_train)

    # --- Random Forest ---
    rf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
                ),
            ),
        ]
    )
    rf_grid = {
        "clf__max_depth": [None, 4, 6, 8, 12],
        "clf__min_samples_split": [2, 4, 6],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    rf_gs = GridSearchCV(rf, rf_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    rf_gs.fit(train_df[FEATURE_COLS], y_train)

    # --- SVM RBF ---
    svm = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
        ]
    )
    svm_grid = {
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    }
    svm_gs = GridSearchCV(svm, svm_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    svm_gs.fit(train_df[FEATURE_COLS], y_train)

    searches = [
        ("logistic_regression", lr_gs),
        ("random_forest", rf_gs),
        ("svm_rbf", svm_gs),
    ]

    rows: list[dict[str, object]] = []
    lines.append("")
    lines.append("optimized_models (train-only selection; dev evaluated once):")
    for name, gs in searches:
        best = gs.best_estimator_
        params = gs.best_params_
        y_pred = best.predict(dev_df[FEATURE_COLS])
        y_score = best.predict_proba(dev_df[FEATURE_COLS])[:, 1]
        m = _metrics(y_dev, y_pred, y_score)
        tn, fp, fn, tp = confusion_matrix(y_dev, y_pred, labels=[0, 1]).ravel()

        base = baseline_map.get(name, {})
        rows.append(
            {
                "model": name,
                "split": "dev",
                **m,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "best_params": str(params),
                "feature_cols": str(FEATURE_COLS),
                "baseline_auroc": base.get("auroc", float("nan")),
                "baseline_accuracy": base.get("accuracy", float("nan")),
                "baseline_f1": base.get("f1", float("nan")),
                "delta_auroc": float(m["auroc"] - base.get("auroc", float("nan"))) if base else float("nan"),
                "delta_accuracy": float(m["accuracy"] - base.get("accuracy", float("nan"))) if base else float("nan"),
                "delta_f1": float(m["f1"] - base.get("f1", float("nan"))) if base else float("nan"),
            }
        )

        lines.append(f"- model: {name}")
        lines.append(f"  best_params: {params}")
        lines.append(f"  dev_metrics: {m}")
        lines.append(f"  confusion(TN,FP,FN,TP): {tn},{fp},{fn},{tp}")
        if base:
            lines.append(
                f"  baseline(auroc/acc/f1): {base.get('auroc')}/{base.get('accuracy')}/{base.get('f1')}"
            )
            lines.append(
                f"  delta(auroc/acc/f1): {m['auroc']-base.get('auroc'):.6g}/{m['accuracy']-base.get('accuracy'):.6g}/{m['f1']-base.get('f1'):.6g}"
            )

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    lines.insert(1, f"output_csv: {out_csv}")
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()

