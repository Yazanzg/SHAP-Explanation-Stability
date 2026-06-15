"""Extract and validate the 13-feature modelling matrix from the cohort CSV.

Config-driven and dataset-agnostic: reads the master table named in config.yaml,
cleans transcripts, and computes exactly the configured features (the resubmission
13; semantic_coherence is excluded). Writes the feature matrix plus a validation
report. It does NOT run modelling, SHAP, or stability analysis.

Outputs:
  <features_csv>                                  (from config.paths.features_csv)
  outputs/process2_feature_validation_report.txt
"""

from __future__ import annotations

import sys
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
    DIAGNOSIS_COLUMN,
    FEATURE_COLUMNS,
    TEST_SPLIT_VALUE,
    TRAIN_SPLIT_VALUE,
)
from src.data_loading import load_dataset_table
from src.feature_extraction import build_feature_matrix


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.FEATURES_CSV
    out_report = config.OUTPUT_DIR / "process2_feature_validation_report.txt"

    lines: list[str] = ["Feature extraction + validation (13-feature modelling matrix)"]

    table = load_dataset_table(load_text=True)
    feats = build_feature_matrix(
        table,
        text_column="transcript",
        id_column=COL_PARTICIPANT_ID,
        label_column=COL_LABEL,
        split_column=COL_SPLIT,
    )
    if DIAGNOSIS_COLUMN in table.columns:
        feats.insert(
            1, DIAGNOSIS_COLUMN, table[DIAGNOSIS_COLUMN].values
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_csv, index=False)

    lines.append(f"output_csv: {out_csv}")
    lines.append(f"rows: {len(feats)}")
    lines.append(f"unique_participant_id: {feats[COL_PARTICIPANT_ID].nunique()}")
    lines.append(f"n_features: {len(FEATURE_COLUMNS)}")
    lines.append(f"features: {list(FEATURE_COLUMNS)}")

    # Split counts (from config expectations)
    split_counts = feats[COL_SPLIT].value_counts().to_dict()
    lines.append(f"split_counts: {split_counts}")
    exp_train = sum(config.EXPECTED_TRAIN_POOL.values())
    exp_test = sum(config.EXPECTED_TEST.values())
    if exp_train and split_counts.get(TRAIN_SPLIT_VALUE) != exp_train:
        raise AssertionError(
            f"Expected {exp_train} {TRAIN_SPLIT_VALUE} rows, got {split_counts.get(TRAIN_SPLIT_VALUE)}"
        )
    if exp_test and split_counts.get(TEST_SPLIT_VALUE) != exp_test:
        raise AssertionError(
            f"Expected {exp_test} {TEST_SPLIT_VALUE} rows, got {split_counts.get(TEST_SPLIT_VALUE)}"
        )

    # Binary label counts
    bin_counts = feats[COL_LABEL].value_counts().sort_index().to_dict()
    lines.append(f"binary_label_counts: {bin_counts}")

    # Diagnosis breakdown (reported, PROCESS-2-specific)
    if DIAGNOSIS_COLUMN in feats.columns:
        lines.append(f"diagnosis_counts: {feats[DIAGNOSIS_COLUMN].value_counts().to_dict()}")

    # Duplicates
    dup = int(feats[COL_PARTICIPANT_ID].duplicated().sum())
    lines.append(f"duplicated_participant_id: {dup}")
    if dup:
        raise AssertionError("Duplicated participant_id values exist")

    # NaN / inf checks on features
    arr = feats[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    lines.append(f"feature_nan_count: {n_nan}")
    lines.append(f"feature_inf_count: {n_inf}")
    if n_inf:
        raise AssertionError("Infinite values present in feature matrix")

    lines.append("")
    lines.append("missing_values_per_feature:")
    for c in FEATURE_COLUMNS:
        lines.append(f"  {c}: {int(feats[c].isna().sum())}")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")
    print(f"rows={len(feats)} features={len(FEATURE_COLUMNS)} nan={n_nan} inf={n_inf}")


if __name__ == "__main__":
    main()
