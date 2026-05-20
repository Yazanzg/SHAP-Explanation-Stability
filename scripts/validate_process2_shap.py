"""Generate SHAP values for PROCESS-2 models only (no stability analysis).

Inputs:
  outputs/process2_features.csv
  outputs/process2_optimized_results.csv (optional; params are also hard-coded per thesis run)

Outputs:
  outputs/shap/process2_shap_values_{baseline|optimized}_{model}.csv
  outputs/process2_shap_validation_report.txt

This script:
  - fits each model on Train split only
  - explains Dev split only (80 participants)
  - uses a fixed background sample from Train for KernelSHAP
  - does NOT compute stability metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
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

EXCLUDE = {
    "semantic_coherence",
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


def _ensure_feature_contract(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required feature columns: {missing}")
    if "semantic_coherence" in df.columns and (df["semantic_coherence"] == 0.0).all():
        # explicitly excluded
        pass

    # Confirm no extras are used
    extras = [c for c in FEATURE_COLS if c in EXCLUDE]
    if extras:
        raise AssertionError(f"Feature list includes excluded columns: {extras}")


def _fit_baseline_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=10000, random_state=RANDOM_SEED)),
            ]
        ),
        "random_forest": Pipeline(
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
        "svm_rbf": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
            ]
        ),
    }


def _fit_optimized_models() -> dict[str, Pipeline]:
    # Best params from validated optimization step
    return {
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=10000, random_state=RANDOM_SEED, C=0.5
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
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
        "svm_rbf": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_SEED,
                        C=5.0,
                        gamma=0.01,
                    ),
                ),
            ]
        ),
    }


def _linear_shap(pipe: Pipeline, X_train: pd.DataFrame, X_dev: pd.DataFrame) -> tuple[np.ndarray, float]:
    # Explain in the transformed (scaled) space but attribute to original features via masker on transformed data.
    Xt_train = pipe[:-1].transform(X_train)
    Xt_dev = pipe[:-1].transform(X_dev)
    clf = pipe[-1]
    masker = shap.maskers.Independent(Xt_train)
    explainer = shap.LinearExplainer(clf, masker)
    sv = explainer(Xt_dev)
    vals = np.asarray(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)) and np.asarray(base).ndim > 0:
        base_val = float(np.asarray(base)[1]) if len(np.asarray(base)) > 1 else float(np.asarray(base)[0])
    else:
        base_val = float(base)
    return vals, base_val


def _tree_shap(pipe: Pipeline, X_train: pd.DataFrame, X_dev: pd.DataFrame) -> tuple[np.ndarray, float]:
    Xt_train = pipe[:-1].transform(X_train)
    Xt_dev = pipe[:-1].transform(X_dev)
    clf = pipe[-1]
    explainer = shap.TreeExplainer(clf, data=Xt_train, feature_names=FEATURE_COLS)
    sv = explainer(Xt_dev)
    vals = np.asarray(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)) and np.asarray(base).ndim > 0:
        base_val = float(np.asarray(base)[1]) if len(np.asarray(base)) > 1 else float(np.asarray(base)[0])
    else:
        base_val = float(base)
    return vals, base_val


def _kernel_shap(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_dev: pd.DataFrame,
    *,
    background_size: int,
    nsamples: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_bg = min(background_size, len(X_train))
    bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
    bg = X_train.iloc[bg_idx]

    def f(X_array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(X_array, columns=FEATURE_COLS)
        return pipe.predict_proba(frame)

    x_dev = X_dev.to_numpy(dtype=float)
    bg_arr = bg.to_numpy(dtype=float)
    explainer = shap.KernelExplainer(f, bg_arr)
    vals = explainer.shap_values(x_dev, nsamples=nsamples, silent=True)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    base = explainer.expected_value
    base_val = float(base[1]) if isinstance(base, (list, np.ndarray)) and len(base) > 1 else float(np.asarray(base).ravel()[0])
    return vals, base_val


def _write_shap_csv(
    out_path: Path,
    *,
    meta: pd.DataFrame,
    shap_vals: np.ndarray,
    expected_value: float,
    model: str,
    stage: str,
) -> None:
    if shap_vals.shape != (len(meta), len(FEATURE_COLS)):
        raise AssertionError(f"SHAP shape mismatch for {model}/{stage}: {shap_vals.shape}")
    out = meta[["participant_id", "split", "binary_label"]].copy()
    out["model"] = model
    out["stage"] = stage
    out["expected_value"] = expected_value
    for j, col in enumerate(FEATURE_COLS):
        out[col] = shap_vals[:, j]
    out.to_csv(out_path, index=False)


def main() -> None:
    in_path = OUTPUT_DIR / "process2_features.csv"
    shap_dir = OUTPUT_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "process2_shap_validation_report.txt"

    df = pd.read_csv(in_path)
    _ensure_feature_contract(df)

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    if len(train_df) != 320 or len(dev_df) != 80:
        raise AssertionError(f"Unexpected split sizes: train={len(train_df)} dev={len(dev_df)}")

    meta_dev = dev_df[["participant_id", "split", "binary_label"]].copy()
    X_train = train_df[FEATURE_COLS]
    X_dev = dev_df[FEATURE_COLS]
    y_train = train_df["binary_label"].to_numpy(dtype=int)

    lines: list[str] = []
    lines.append(f"input_csv: {in_path}")
    lines.append(f"train_rows: {len(train_df)}")
    lines.append(f"dev_rows: {len(dev_df)}")
    lines.append(f"feature_cols_used(13): {FEATURE_COLS}")
    lines.append(f"kernelshap_background_size: {min(SHAP_BACKGROUND_SIZE, len(train_df))}")
    lines.append(f"kernelshap_nsamples: {KERNEL_SHAP_NSAMPLES}")
    lines.append("")

    outputs: list[Path] = []

    # Baseline models
    baseline = _fit_baseline_models()
    optimized = _fit_optimized_models()

    for stage, models in [("baseline", baseline), ("optimized", optimized)]:
        for model_name, pipe in models.items():
            pipe.fit(X_train, y_train)
            if model_name == "logistic_regression":
                vals, base = _linear_shap(pipe, X_train, X_dev)
            elif model_name == "random_forest":
                vals, base = _tree_shap(pipe, X_train, X_dev)
            elif model_name == "svm_rbf":
                # KernelSHAP retry ladder for reliability
                last_exc: Exception | None = None
                for ns in [KERNEL_SHAP_NSAMPLES, max(40, KERNEL_SHAP_NSAMPLES // 2), 20]:
                    try:
                        vals, base = _kernel_shap(
                            pipe,
                            X_train,
                            X_dev,
                            background_size=SHAP_BACKGROUND_SIZE,
                            nsamples=ns,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        vals = None  # type: ignore[assignment]
                        base = float("nan")
                else:
                    raise RuntimeError(f"KernelSHAP failed for {stage}/{model_name}") from last_exc
            else:
                raise ValueError(model_name)

            if not np.isfinite(vals).all():
                raise AssertionError(f"Non-finite SHAP values for {stage}/{model_name}")

            out_path = shap_dir / f"process2_shap_values_{stage}_{model_name}.csv"
            _write_shap_csv(
                out_path,
                meta=meta_dev,
                shap_vals=vals,
                expected_value=base,
                model=model_name,
                stage=stage,
            )
            outputs.append(out_path)

            mean_abs = np.mean(np.abs(vals), axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]
            lines.append(f"{stage}/{model_name}: wrote {out_path.name}")
            lines.append(f"  expected_value: {base}")
            lines.append("  top10_mean_abs_shap:")
            for i in top_idx:
                lines.append(f"    {FEATURE_COLS[i]}: {mean_abs[i]:.6g}")
            lines.append("")

    # Validation: files present and shape checks
    lines.append("validation:")
    lines.append(f"shap_files_written: {len(outputs)} (expected 6)")
    if len(outputs) != 6:
        raise AssertionError("Did not write 6 SHAP files")

    for p in outputs:
        d = pd.read_csv(p)
        # dev rows = 80
        if len(d) != 80:
            raise AssertionError(f"{p.name} expected 80 rows, got {len(d)}")
        # columns include metadata + 13 features
        if any(c not in d.columns for c in ["participant_id", "split", "binary_label", "model", "stage", "expected_value"]):
            raise AssertionError(f"{p.name} missing required metadata columns")
        feat_cols_present = [c for c in FEATURE_COLS if c in d.columns]
        if feat_cols_present != FEATURE_COLS:
            raise AssertionError(f"{p.name} feature columns mismatch")
        if "semantic_coherence" in d.columns:
            raise AssertionError(f"{p.name} should not include semantic_coherence")
        if not np.isfinite(d[FEATURE_COLS].to_numpy(dtype=float)).all():
            raise AssertionError(f"{p.name} contains non-finite SHAP values")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")
    for p in outputs:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()

