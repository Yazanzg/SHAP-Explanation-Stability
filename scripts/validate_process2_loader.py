"""Validate PROCESS-2 loader outputs without running preprocessing/features/models.

Writes: outputs/process2_loader_validation_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, PROCESS2_MASTER_CSV
from src.data_loading import load_process2_master_table


def _assert_eq(name: str, got: int, expected: int, lines: list[str]) -> None:
    ok = got == expected
    lines.append(f"{name}: {got} (expected {expected}) -> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"{name} expected {expected}, got {got}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "process2_loader_validation_report.txt"

    lines: list[str] = []
    lines.append(f"master_csv: {PROCESS2_MASTER_CSV}")
    lines.append(f"master_exists: {PROCESS2_MASTER_CSV.is_file()}")

    df = load_process2_master_table(load_text=True)

    # Core shape
    _assert_eq("rows", len(df), 400, lines)
    _assert_eq("unique_participant_id", df["participant_id"].nunique(), 400, lines)

    # Required columns
    required_cols = [
        "participant_id",
        "split",
        "diagnosis",
        "binary_label",
        "transcript_path",
        "transcript",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    lines.append(f"missing_required_columns: {missing}")
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")

    # Local transcript path exists
    tp = df["transcript_path"].astype(str).map(Path)
    exists = tp.map(lambda p: p.is_file())
    lines.append(f"transcripts_local_existing: {int(exists.sum())} / {len(exists)}")
    if not bool(exists.all()):
        bad = df.loc[~exists, "participant_id"].head(10).tolist()
        raise AssertionError(f"Missing transcript files for participants: {bad}")

    # Non-empty transcript strings
    t = df["transcript"].astype(str)
    nonempty = t.map(lambda s: bool(s.strip()))
    lines.append(f"nonempty_transcripts: {int(nonempty.sum())} / {len(nonempty)}")
    if not bool(nonempty.all()):
        bad = df.loc[~nonempty, "participant_id"].head(10).tolist()
        raise AssertionError(f"Empty transcript strings for participants: {bad}")

    # Counts
    diag = df["diagnosis"].value_counts().to_dict()
    lines.append(f"diagnosis_counts: {diag}")
    if diag.get("HC") != 200 or diag.get("MCI") != 150 or diag.get("Dementia") != 50:
        raise AssertionError(f"Unexpected diagnosis distribution: {diag}")

    bin_counts = df["binary_label"].value_counts().sort_index().to_dict()
    lines.append(f"binary_counts: {bin_counts}")
    if bin_counts.get(0) != 200 or bin_counts.get(1) != 200:
        raise AssertionError(f"Unexpected binary distribution: {bin_counts}")

    split_counts = df["split"].value_counts().to_dict()
    lines.append(f"split_counts: {split_counts}")
    if split_counts.get("train") != 320 or split_counts.get("dev") != 80:
        raise AssertionError(f"Unexpected split distribution: {split_counts}")

    # Preview first 3 rows
    lines.append("")
    lines.append("first_3_rows_preview:")
    for _, r in df.head(3).iterrows():
        txt = str(r["transcript"])
        lines.append(
            " - "
            + " | ".join(
                [
                    f"participant_id={r['participant_id']}",
                    f"split={r['split']}",
                    f"diagnosis={r['diagnosis']}",
                    f"binary_label={int(r['binary_label'])}",
                    f"len={len(txt)}",
                    f"head200={txt[:200].replace('\\n',' ')}",
                ]
            )
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Print first 3 rows in the requested format
    preview = df.head(3).copy()
    preview["transcript_len"] = preview["transcript"].astype(str).map(len)
    preview["transcript_head200"] = preview["transcript"].astype(str).map(lambda s: s[:200].replace("\n", " "))
    print(preview[["participant_id", "split", "diagnosis", "binary_label", "transcript_len", "transcript_head200"]].to_string(index=False))
    print(f"\nWrote: {report_path}")


if __name__ == "__main__":
    main()

