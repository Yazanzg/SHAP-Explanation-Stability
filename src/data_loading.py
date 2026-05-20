"""Discover dataset layout and load participant-level labels / PROCESS-2 cohort."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATA_DIR,
    DATASET_NAME,
    LABEL_MODE,
    PROCESS2_MASTER_CSV,
    PROCESS2_EXPECTED_CI,
    PROCESS2_EXPECTED_HC,
    PROCESS2_METADATA_FILE,
    PROJECT_ROOT,
    is_process2,
)

logger = logging.getLogger(__name__)

_PROCESS2_ID_PATTERN = re.compile(r"PROCESS-2_rec__\d+", re.I)


def print_directory_tree(root: Path, max_depth: int = 3, max_files: int = 80) -> None:
    """Print a bounded listing of files under root for debugging."""
    import os

    print(f"\n--- Discovered dataset root: {root} ---")
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        rel = Path(dirpath).relative_to(root)
        print(f"[dir] {rel}")
        for fn in sorted(filenames):
            if count >= max_files:
                print("... (listing truncated; increase max_files if needed)")
                return
            print(f"  {fn}")
            count += 1
        if count >= max_files:
            print("... (listing truncated)")
            return


def _find_augmented_metadata(data_dir: Path) -> Path | None:
    candidates = list(data_dir.glob("Data_AUG*.csv")) + list(data_dir.glob("*output*.csv"))
    for c in candidates:
        if c.is_file() and "Data_AUG" in c.name:
            return c
    return None


def _find_train_transcript_table(data_dir: Path) -> Path | None:
    p = data_dir / "df_train.csv"
    return p if p.is_file() else None


def _resolve_participant_id_column(df: pd.DataFrame) -> str:
    for c in (
        "participant_id",
        "Participant_ID",
        "participant",
        "ID",
        "Record-ID",
        "rec_id",
    ):
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].astype(str).str.contains("PROCESS-2_rec", na=False).any():
            return c
    raise ValueError(
        "Could not find participant ID column in metadata.csv. "
        f"Columns present: {list(df.columns)}"
    )


def _normalize_split(value: object) -> str:
    v = str(value).strip().lower()
    if v in ("train", "training", "tr"):
        return "train"
    if v in ("test", "testing", "te", "dev"):
        return "test"
    raise ValueError(f"Unknown Split value: {value!r} (expected Train or Test)")


def map_diagnosis_to_binary(diagnosis: str) -> int:
    """HC=0; MCI and Dementia=1 (cognitive impairment positive class)."""
    d = str(diagnosis).strip()
    if d == "HC":
        return 0
    if d in ("MCI", "Dementia"):
        return 1
    raise ValueError(f"Unknown diagnosis label: {diagnosis!r}")


def load_process2_metadata(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load PROCESS-2 ``metadata.csv`` with binary labels and predefined split.

    Returns columns: participant_id, diagnosis, diagnosis_label, split.
    """
    root = data_dir or DATA_DIR
    meta_path = root / PROCESS2_METADATA_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"PROCESS-2 metadata not found: {meta_path}")

    df = pd.read_csv(meta_path)
    id_col = _resolve_participant_id_column(df)
    if "diagnosis" not in df.columns:
        raise ValueError(f"{meta_path} must contain column 'diagnosis'")
    if "Split" not in df.columns:
        raise ValueError(f"{meta_path} must contain column 'Split'")

    out = pd.DataFrame()
    out["participant_id"] = df[id_col].astype(str).str.strip()
    out["diagnosis"] = df["diagnosis"].astype(str).str.strip()
    out["diagnosis_label"] = out["diagnosis"].map(map_diagnosis_to_binary).astype(int)
    out["split"] = df["Split"].map(_normalize_split)

    dup = out["participant_id"].duplicated().sum()
    if dup:
        raise ValueError(f"Duplicate participant_id rows in metadata: {dup}")

    if out["diagnosis_label"].isna().any():
        raise ValueError("Missing or unmapped diagnosis labels in metadata.")

    logger.info("Loaded PROCESS-2 metadata: %s (%s participants)", meta_path, len(out))
    return out.reset_index(drop=True)


def load_process2_master_table(master_csv: Path | None = None, *, load_text: bool = False) -> pd.DataFrame:
    """
    Load the authoritative PROCESS-2 CTD-only master table.

    Expected columns (minimum):
    - participant_id
    - transcript_path (local .txt)
    - diagnosis (HC/MCI/Dementia)
    - binary_label (0=HC, 1=CI)
    - split (train/dev or train/test)
    """
    p = master_csv or PROCESS2_MASTER_CSV
    if not Path(p).is_file():
        raise FileNotFoundError(f"PROCESS-2 master CSV not found: {p}")
    df = pd.read_csv(p)
    required = {"participant_id", "transcript_path", "diagnosis", "binary_label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {sorted(missing)}")
    out = df.copy()
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["diagnosis"] = out["diagnosis"].astype(str).str.strip()
    out["diagnosis_label"] = out["binary_label"].astype(int)
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    # Normalize split naming: allow dev as test-equivalent
    out.loc[out["split"] == "test", "split"] = "dev"
    dup = out["participant_id"].duplicated().sum()
    if dup:
        raise ValueError(f"Duplicate participant_id rows in master CSV: {dup}")

    if load_text:
        texts: list[str] = []
        for path in out["transcript_path"].astype(str).tolist():
            pth = Path(path)
            try:
                txt = pth.read_text(encoding="utf-8", errors="replace")
            except OSError:
                txt = ""
            texts.append(txt)
        out["transcript"] = pd.Series(texts, index=out.index).astype(str)
    return out.reset_index(drop=True)


def validate_process2_master(master: pd.DataFrame) -> None:
    """Validate master table has local CTD transcripts and expected counts."""
    # Transcript locality
    tp = master["transcript_path"].astype(str)
    missing = []
    for i, path in enumerate(tp):
        p = Path(path)
        if not p.is_file():
            missing.append(master["participant_id"].iloc[i])
    if missing:
        raise FileNotFoundError(
            f"Missing local CTD transcript files for {len(missing)} participants "
            f"(e.g. {missing[:5]})."
        )

    print("\n--- PROCESS-2 master diagnosis (3-class) ---")
    print(master["diagnosis"].value_counts().to_string())
    print("\n--- PROCESS-2 master binary (HC=0, CI=1) ---")
    print(master["diagnosis_label"].value_counts().sort_index().to_string())
    print("\n--- PROCESS-2 master split ---")
    print(master["split"].value_counts().to_string())

    n_hc = int((master["diagnosis_label"] == 0).sum())
    n_ci = int((master["diagnosis_label"] == 1).sum())
    if n_hc != PROCESS2_EXPECTED_HC or n_ci != PROCESS2_EXPECTED_CI:
        logger.warning(
            "Binary class counts (%s HC, %s CI) differ from expected (%s HC, %s CI).",
            n_hc,
            n_ci,
            PROCESS2_EXPECTED_HC,
            PROCESS2_EXPECTED_CI,
        )


def validate_process2(data_dir: Path | None = None, meta: pd.DataFrame | None = None) -> None:
    """Startup checks for PROCESS-2 layout, labels, CTD files, and balance."""
    root = data_dir or DATA_DIR
    # Prefer the authoritative master table if present; otherwise fall back to metadata.csv
    if meta is None:
        try:
            meta = load_process2_master_table()
        except FileNotFoundError:
            meta = load_process2_metadata(root)
    # Master CSV has transcript paths; metadata.csv does not
    if "transcript_path" in meta.columns:
        validate_process2_master(meta)
        return

    missing_ctd: list[str] = []
    for pid in meta["participant_id"]:
        if load_process2_ctd_path(pid, root) is None:
            missing_ctd.append(pid)
    if missing_ctd:
        raise FileNotFoundError(
            f"Missing CTD transcript for {len(missing_ctd)} participants "
            f"(e.g. {missing_ctd[:5]}). Expected "
            f"{{root}}/{{id}}/{{id}}__CTD.txt"
        )

    print("\n--- PROCESS-2 diagnosis (3-class) ---")
    print(meta["diagnosis"].value_counts().to_string())
    print("\n--- PROCESS-2 binary (HC=0, CI=MCI+Dementia=1) ---")
    print(meta["diagnosis_label"].value_counts().sort_index().to_string())
    print("\n--- PROCESS-2 predefined split ---")
    print(meta.groupby("split")["diagnosis_label"].value_counts().to_string())

    n_hc = int((meta["diagnosis_label"] == 0).sum())
    n_ci = int((meta["diagnosis_label"] == 1).sum())
    if n_hc != PROCESS2_EXPECTED_HC or n_ci != PROCESS2_EXPECTED_CI:
        logger.warning(
            "Binary class counts (%s HC, %s CI) differ from expected (%s HC, %s CI).",
            n_hc,
            n_ci,
            PROCESS2_EXPECTED_HC,
            PROCESS2_EXPECTED_CI,
        )

    n_train = int((meta["split"] == "train").sum())
    n_test = int((meta["split"] == "test").sum())
    print(f"\nTrain: {n_train}  |  Test: {n_test}")
    if n_train == 0 or n_test == 0:
        raise ValueError("Train or Test split is empty — check Split column in metadata.csv.")


def load_process2_ctd_path(participant_id: str, data_dir: Path | None = None) -> Path | None:
    """Return path to ``*__CTD.txt`` for a participant, or None if missing."""
    root = data_dir or DATA_DIR
    pid = str(participant_id).strip()
    folder = root / pid
    if folder.is_dir():
        matches = sorted(folder.glob(f"{pid}__CTD.txt")) + sorted(folder.glob("*__CTD.txt"))
        for p in matches:
            if p.is_file() and p.name.endswith("__CTD.txt"):
                return p
    # flat fallback
    for p in root.glob(f"{pid}__CTD.txt"):
        if p.is_file():
            return p
    return None


def load_cohort_table(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load full cohort table for the active dataset mode.

    PROCESS-2: participant_id, diagnosis, diagnosis_label, split.
    Legacy: participant_id, diagnosis_label only.
    """
    if is_process2():
        # Prefer master table (CTD-only + split + paths). If missing, fall back.
        try:
            return load_process2_master_table()
        except FileNotFoundError:
            return load_process2_metadata(data_dir)
    labels = load_label_table(data_dir)
    if "split" not in labels.columns:
        labels["split"] = "train"
    if "diagnosis" not in labels.columns:
        labels["diagnosis"] = labels["diagnosis_label"].map({0: "HC", 1: "CI"})
    return labels


def load_label_table(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load participant_id and diagnosis_label (0=HC/control, 1=CI positive).

    PROCESS-2 mode uses ``metadata.csv``; legacy mode uses participant_labels.csv
    or Data_AUG*.csv (development fallback).
    """
    if is_process2():
        meta = load_process2_metadata(data_dir)
        return meta[["participant_id", "diagnosis_label"]].copy()

    root = data_dir or DATA_DIR
    custom = root / "participant_labels.csv"
    if custom.is_file():
        df = pd.read_csv(custom)
        required = {"participant_id", "diagnosis_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{custom} must contain columns {sorted(required)}; missing {sorted(missing)}"
            )
        logger.info("Loaded labels from %s (legacy participant_labels.csv).", custom)
        return df[["participant_id", "diagnosis_label"]].copy()

    aug = _find_augmented_metadata(root)
    if aug is not None:
        df = pd.read_csv(aug)
        if "Record-ID" not in df.columns:
            raise ValueError(f"Expected 'Record-ID' in {aug}")
        out = pd.DataFrame({"participant_id": df["Record-ID"].astype(str)})
        if "Class" in df.columns:
            cls = df["Class"].astype(str)
            if LABEL_MODE == "binary_hc_dementia":
                mask = cls.isin(["HC", "Dementia"])
                dropped = int((~mask).sum())
                out["diagnosis_label"] = cls.map({"HC": 0, "Dementia": 1})
                out = out.loc[mask].copy()
                out["diagnosis_label"] = out["diagnosis_label"].astype(int)
                if dropped:
                    logger.warning(
                        "LABEL_MODE=%s: dropped %s rows not in {HC, Dementia}.",
                        LABEL_MODE,
                        dropped,
                    )
            elif LABEL_MODE == "binary_hc_pathological":
                out["diagnosis_label"] = cls.map(
                    {"HC": 0, "MCI": 1, "Dementia": 1}
                )
                out = out[cls.isin(["HC", "MCI", "Dementia"])].copy()
                out["diagnosis_label"] = out["diagnosis_label"].astype(int)
            else:
                raise ValueError(f"Unknown LABEL_MODE: {LABEL_MODE}")
            logger.warning(
                "LEGACY development labels from %s — not PROCESS-2 final cohort.",
                aug,
            )
            return out.reset_index(drop=True)

    raise FileNotFoundError(
        "Could not find labels. For PROCESS-2 set THESIS_DATASET=PROCESS2 and "
        f"provide {root / PROCESS2_METADATA_FILE}. For legacy, use participant_labels.csv "
        "or Data_AUG_*.csv."
    )


def train_test_masks(cohort: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Boolean masks for train and test rows from cohort ``split`` column."""
    if "split" not in cohort.columns:
        raise ValueError("Cohort table missing 'split' column.")
    train_m = cohort["split"].astype(str).str.lower() == "train"
    test_m = cohort["split"].astype(str).str.lower() == "test"
    if train_m.sum() == 0 or test_m.sum() == 0:
        raise ValueError("Train or Test mask is empty.")
    overlap = (train_m & test_m).sum()
    if overlap:
        raise ValueError("Participants cannot be in both Train and Test.")
    return train_m, test_m


def discover_transcript_roots(data_dir: Path | None = None) -> list[Path]:
    root = data_dir or DATA_DIR
    if is_process2():
        return [root.resolve()]
    roots: list[Path] = []
    archive = root / "Archive"
    if archive.is_dir():
        roots.append(archive)
    for p in root.rglob("*__CTD.txt"):
        roots.append(p.parent.parent)
        break
    if not roots and root.is_dir():
        roots.append(root)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        r = r.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def summarize_cohort(labels: pd.DataFrame, *, full_meta: pd.DataFrame | None = None) -> None:
    """Print label and split summaries."""
    meta = full_meta if full_meta is not None else labels
    print("\n--- Label summary (binary) ---")
    print(labels["diagnosis_label"].value_counts().sort_index().to_string())
    if "diagnosis" in meta.columns:
        print("\n--- Original diagnosis ---")
        print(meta["diagnosis"].value_counts().to_string())
    if "split" in meta.columns:
        print("\n--- Split ---")
        print(meta["split"].value_counts().to_string())
    dup = labels["participant_id"].duplicated().sum()
    if dup:
        raise ValueError(f"Found {dup} duplicated participant_id rows in labels.")


def validate_paths() -> dict[str, Any]:
    root = DATA_DIR
    info: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(root),
        "dataset_name": DATASET_NAME,
        "is_process2": is_process2(),
    }
    if is_process2():
        info["metadata_csv"] = str(root / PROCESS2_METADATA_FILE)
        info["has_metadata"] = (root / PROCESS2_METADATA_FILE).is_file()
        info["label_source"] = "PROCESS-2 metadata.csv"
    else:
        labels_csv = root / "participant_labels.csv"
        info["has_archive"] = (root / "Archive").is_dir()
        info["df_train"] = str(_find_train_transcript_table(root) or "")
        info["augmented_csv"] = str(_find_augmented_metadata(root) or "")
        info["has_participant_labels_csv"] = labels_csv.is_file()
        info["label_source"] = (
            "participant_labels.csv"
            if labels_csv.is_file()
            else ("Data_AUG*.csv (legacy)" if _find_augmented_metadata(root) else "none")
        )
    return info
