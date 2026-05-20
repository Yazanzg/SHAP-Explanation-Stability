"""Extract and validate PROCESS-2 linguistic features (CTD) only.

This script:
- loads PROCESS-2 master table (with raw transcript text)
- cleans transcript text
- extracts linguistic features from cleaned text only
- writes outputs/process2_features.csv
- writes outputs/process2_feature_validation_report.txt

It does NOT run modeling, optimization, SHAP, or stability analysis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR
from src.data_loading import load_process2_master_table
from src.feature_extraction import extract_row_features
from src.preprocessing import clean_transcript


def _wc(text: str) -> int:
    t = str(text).strip()
    return 0 if not t else len(t.split())


def _sentence_count(text: str) -> int:
    # Keep it lightweight; coherence/POS already use spaCy downstream.
    t = str(text).strip()
    if not t:
        return 0
    # crude split on punctuation; robust enough for a control feature
    parts = [p for p in re_split_sentence(t) if p.strip()]
    return len(parts)


def re_split_sentence(t: str) -> list[str]:
    import re

    return re.split(r"[.!?]+", t)


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "process2_features.csv"
    out_report = OUTPUT_DIR / "process2_feature_validation_report.txt"

    lines: list[str] = []
    lines.append("PROCESS-2 feature extraction + validation")

    df = load_process2_master_table(load_text=True)

    # Preprocess (clean) without re-reading any files
    df["transcript"] = df["transcript"].astype(str)
    df["cleaned_transcript"] = df["transcript"].map(clean_transcript).astype(str)

    df["word_count"] = df["cleaned_transcript"].map(_wc).astype(int)
    df["sentence_count"] = df["transcript"].map(_sentence_count).astype(int)

    # Feature extraction from cleaned text only
    feats = []
    for txt in df["cleaned_transcript"].tolist():
        feats.append(extract_row_features(str(txt)))
    feat_df = pd.DataFrame(feats)

    # Assemble output table (preserve requested columns)
    keep = [
        "participant_id",
        "diagnosis",
        "diagnosis_label",
        "binary_label",
        "split",
        "transcript",
        "cleaned_transcript",
        "word_count",
        "sentence_count",
    ]
    out = df[keep].merge(
        feat_df,
        left_index=True,
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    out.to_csv(out_csv, index=False)

    # -----------------------
    # Validation checks
    # -----------------------
    lines.append(f"wrote_csv: {out_csv}")
    lines.append(f"rows: {len(out)}")
    lines.append(f"unique_participant_id: {out['participant_id'].nunique()}")

    def assert_eq(label: str, got: int, expected: int) -> None:
        ok = got == expected
        lines.append(f"{label}: {got} (expected {expected}) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(f"{label} expected {expected}, got {got}")

    assert_eq("rows", len(out), 400)
    assert_eq("unique_participant_id", out["participant_id"].nunique(), 400)

    # Missing targets / split
    miss_bin = int(out["binary_label"].isna().sum())
    miss_split = int(out["split"].isna().sum())
    lines.append(f"missing_binary_label: {miss_bin}")
    lines.append(f"missing_split: {miss_split}")
    if miss_bin:
        raise AssertionError("binary_label has missing values")
    if miss_split:
        raise AssertionError("split has missing values")

    # Split counts
    split_counts = out["split"].value_counts().to_dict()
    lines.append(f"split_counts: {split_counts}")
    if split_counts.get("train") != 320 or split_counts.get("dev") != 80:
        raise AssertionError(f"Unexpected split counts: {split_counts}")

    # Binary counts
    bin_counts = out["binary_label"].value_counts().sort_index().to_dict()
    lines.append(f"binary_label_counts: {bin_counts}")
    if bin_counts.get(0) != 200 or bin_counts.get(1) != 200:
        raise AssertionError(f"Unexpected binary counts: {bin_counts}")

    # Diagnosis counts
    diag_counts = out["diagnosis"].value_counts().to_dict()
    lines.append(f"diagnosis_counts: {diag_counts}")
    if diag_counts.get("HC") != 200 or diag_counts.get("MCI") != 150 or diag_counts.get("Dementia") != 50:
        raise AssertionError(f"Unexpected diagnosis counts: {diag_counts}")

    # Duplicates
    dup = int(out["participant_id"].duplicated().sum())
    lines.append(f"duplicated_participant_id: {dup}")
    if dup:
        raise AssertionError("Duplicated participant_id values exist")

    # Feature columns are all numeric columns excluding raw text-like fields
    non_feature = {
        "participant_id",
        "diagnosis",
        "split",
        "transcript",
        "cleaned_transcript",
    }
    numeric_cols = [
        c
        for c in out.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(out[c])
    ]
    lines.append(f"numeric_columns_count: {len(numeric_cols)}")
    lines.append(f"numeric_columns: {numeric_cols}")

    # No fully empty feature rows (all NaN in numeric cols)
    all_nan = out[numeric_cols].isna().all(axis=1)
    n_all_nan = int(all_nan.sum())
    lines.append(f"fully_empty_numeric_rows(all_nan): {n_all_nan}")
    if n_all_nan:
        bad = out.loc[all_nan, "participant_id"].head(10).tolist()
        raise AssertionError(f"Fully empty numeric rows for participants: {bad}")

    # No infinite values
    arr = out[numeric_cols].to_numpy(dtype=float)
    inf_mask = ~np.isfinite(arr)
    n_inf = int(inf_mask.sum())
    lines.append(f"nonfinite_values_count(nan/inf): {n_inf}")
    if n_inf:
        # allow NaN but forbid inf
        n_inf_only = int(np.isinf(arr).sum())
        lines.append(f"infinite_values_count: {n_inf_only}")
        if n_inf_only:
            raise AssertionError("Infinite values present in feature table")

    # Missing-value counts per feature
    lines.append("")
    lines.append("missing_values_per_numeric_column:")
    for c in numeric_cols:
        lines.append(f"  {c}: {int(out[c].isna().sum())}")

    # Descriptive stats
    lines.append("")
    lines.append("descriptive_statistics (numeric columns):")
    desc = out[numeric_cols].describe(include="all").T
    # ensure stable order
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    for c in desc.index.tolist():
        r = desc.loc[c]
        lines.append(
            f"  {c}: count={r['count']:.0f} mean={r['mean']:.6g} std={r['std']:.6g} "
            f"min={r['min']:.6g} p25={r['25%']:.6g} median={r['50%']:.6g} p75={r['75%']:.6g} max={r['max']:.6g}"
        )

    # Preview 5 rows (selected features)
    sel_feats = [
        "word_count",
        "type_token_ratio",
        "filler_count",
        "filler_ratio",
        "mean_clause_length",
        "semantic_coherence",
        "content_density",
    ]
    available_sel = [c for c in sel_feats if c in out.columns]
    preview = out.sample(n=5, random_state=42)[
        ["participant_id", "diagnosis", "binary_label", "split", *available_sel]
    ].reset_index(drop=True)
    lines.append("")
    lines.append("preview_rows (5):")
    lines.append(preview.to_string(index=False))

    _write_report(out_report, lines)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()

