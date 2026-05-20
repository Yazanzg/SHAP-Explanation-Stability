"""Validate PROCESS-2 preprocessing (cleaning) without running feature extraction or models.

Reads raw transcript text from the PROCESS-2 master loader and applies
`src.preprocessing.clean_transcript` to produce `cleaned_transcript`.

Writes: outputs/process2_preprocessing_validation_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR
from src.data_loading import load_process2_master_table
from src.preprocessing import clean_transcript


def _summary_stats(x: pd.Series) -> dict[str, float]:
    arr = x.to_numpy(dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "process2_preprocessing_validation_report.txt"

    df = load_process2_master_table(load_text=True)

    # Preserve participant-level columns
    keep_cols = [
        "participant_id",
        "diagnosis",
        "binary_label",
        "diagnosis_label",
        "split",
        "transcript_path",
        "audio_path",
        "transcript",
    ]
    for c in keep_cols:
        if c not in df.columns:
            raise AssertionError(f"Loader missing required column: {c}")
    out = df[keep_cols].copy()

    out["transcript"] = out["transcript"].astype(str)
    out["cleaned_transcript"] = out["transcript"].map(clean_transcript).astype(str)

    out["raw_length"] = out["transcript"].map(len).astype(int)
    out["cleaned_length"] = out["cleaned_transcript"].map(len).astype(int)

    # Checks
    lines: list[str] = []
    lines.append(f"rows: {len(out)}")
    lines.append(f"unique_participant_id: {out['participant_id'].nunique()}")

    if len(out) != 400:
        raise AssertionError(f"Expected 400 rows, got {len(out)}")
    if out["participant_id"].nunique() != 400:
        raise AssertionError("participant_id is not unique")

    raw_nonempty = out["transcript"].map(lambda s: bool(str(s).strip()))
    cleaned_nonempty = out["cleaned_transcript"].map(lambda s: bool(str(s).strip()))
    lines.append(f"nonempty_raw_transcripts: {int(raw_nonempty.sum())} / {len(raw_nonempty)}")
    lines.append(
        f"nonempty_cleaned_transcripts: {int(cleaned_nonempty.sum())} / {len(cleaned_nonempty)}"
    )
    if not bool(raw_nonempty.all()):
        bad = out.loc[~raw_nonempty, "participant_id"].head(10).tolist()
        raise AssertionError(f"Empty raw transcripts for: {bad}")
    if not bool(cleaned_nonempty.all()):
        bad = out.loc[~cleaned_nonempty, "participant_id"].head(10).tolist()
        raise AssertionError(f"Empty cleaned transcripts for: {bad}")

    # Length stats
    raw_stats = _summary_stats(out["raw_length"])
    cleaned_stats = _summary_stats(out["cleaned_length"])
    lines.append("")
    lines.append("raw_length_stats:")
    for k, v in raw_stats.items():
        lines.append(f"  {k}: {v:.2f}")
    lines.append("cleaned_length_stats:")
    for k, v in cleaned_stats.items():
        lines.append(f"  {k}: {v:.2f}")

    # Preview examples (5)
    lines.append("")
    lines.append("preview_examples (5):")
    sample = out.sample(n=5, random_state=42).reset_index(drop=True)
    for _, r in sample.iterrows():
        raw_head = r["transcript"][:200].replace("\n", " ")
        clean_head = r["cleaned_transcript"][:200].replace("\n", " ")
        lines.append(
            f"- participant_id={r['participant_id']} | diagnosis={r['diagnosis']} | "
            f"binary_label={int(r['binary_label'])} | split={r['split']} | "
            f"raw_len={int(r['raw_length'])} | cleaned_len={int(r['cleaned_length'])}"
        )
        lines.append(f"  raw_head200: {raw_head}")
        lines.append(f"  cleaned_head200: {clean_head}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()

