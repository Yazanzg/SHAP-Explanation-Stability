import csv
from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = Path("c:/Users/Yazan/Thesis")
ARCHIVE = ROOT / "data_old" / "Archive"
METADATA_CANDIDATES = [ROOT / "data_old" / "Data_AUG_13.11.2024_output.csv", ROOT / "data_old" / "metadata.csv"]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
OUT_CSV = OUTPUTS / "process2_ctd_binary_preview.csv"


def find_non_audio_files():
    print("\n=== Non-audio files under Archive ===")
    files = []
    for p in ARCHIVE.rglob("*"):
        if p.is_file() and p.suffix.lower() != ".wav":
            rel = p.relative_to(ROOT)
            files.append(rel)
    for f in sorted(files):
        print(f)
    return files


def preview_metadata_files():
    print("\n=== Preview candidate metadata files ===")
    previews = {}
    for cand in METADATA_CANDIDATES:
        if cand.exists():
            print(f"\nPreviewing {cand}:")
            try:
                df = pd.read_csv(cand)
                print(df.head(10).to_string(index=False))
                previews[str(cand)] = df
            except Exception as e:
                print(f"Failed to read {cand}: {e}")
    # also look for any other csv/tsv/json in data_old
    for p in (ROOT / "data_old").glob("*.*"):
        if p.suffix.lower() in {".csv", ".tsv", ".json", ".txt"} and str(p) not in previews:
            print(f"\nPreviewing additional candidate {p}:")
            try:
                if p.suffix.lower() in {".csv", ".tsv"}:
                    df = pd.read_csv(p, sep=None, engine='python')
                    print(df.head(10).to_string(index=False))
                    previews[str(p)] = df
                else:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f):
                            print(line.rstrip())
                            if i >= 9:
                                break
            except Exception as e:
                print(f"Failed to read {p}: {e}")
    return previews


def identify_diagnosis_column(df):
    # Look for columns containing diagnosis-like names
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["diagn", "class", "group", "label", "dx"]) ]
    print("\nDiagnosis-like columns found:", candidates)
    for c in candidates:
        unique = df[c].dropna().unique()
        print(f"Column {c}: unique values -> {unique[:10]} (total {len(unique)})")
    # heuristic: column with 3 unique values matching HC/MCI/Dementia
    for c in candidates:
        vals = df[c].dropna().unique()
        sval = set(str(x).strip().upper() for x in vals)
        if {'HC','MCI','DEMENTIA'}.issubset(sval) or len(sval & {'HC','MCI','DEMENTIA'})>=2:
            print(f"Selected diagnosis column: {c}")
            return c
    # fallback: check common names
    for name in ["Class","class","Diagnosis","diagnosis","group"]:
        if name in df.columns:
            print(f"Fallback selected: {name}")
            return name
    return None


def build_ctd_table(df_meta, diagnosis_col):
    # df_meta expected to have Record-ID or similar identifying participant folder
    id_col = None
    for cand in ["Record-ID","record-id","participant_id","participant","id"]:
        for c in df_meta.columns:
            if c.lower() == cand.lower():
                id_col = c
                break
        if id_col:
            break
    if id_col is None:
        raise ValueError("Could not find participant ID column in metadata")
    print(f"Using participant id column: {id_col}")

    rows = []
    for folder in sorted(ARCHIVE.iterdir()):
        if folder.is_dir():
            pid = folder.name  # e.g., Process-rec-001
            # participant id in metadata appears as Process-rec-001 or Process-rec-001? check
            # find CTD wav and txt
            ct_wav = None
            ct_txt = None
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() == '.wav' and '__CTD' in f.name:
                    ct_wav = str(f.resolve())
                if f.is_file() and f.suffix.lower() == '.txt' and '__CTD' in f.name:
                    ct_txt = str(f.resolve())
            if not ct_wav and not ct_txt:
                continue
            # lookup metadata row
            # metadata might use 'Process-rec-001' exact or 'Process-rec-001' -> try exact match
            meta_row = df_meta[df_meta[id_col] == pid]
            if meta_row.empty:
                # try without case, or with different separators
                meta_row = df_meta[df_meta[id_col].str.lower() == pid.lower()]
            if meta_row.empty:
                # try matching by numeric suffix
                num = ''.join(ch for ch in pid if ch.isdigit())
                meta_row = df_meta[df_meta[id_col].astype(str).str.contains(num, na=False)]
            diagnosis = None
            split = None
            if not meta_row.empty:
                diagnosis = str(meta_row.iloc[0][diagnosis_col]).strip()
                # split mapping
                for cand in ['TrainOrDev','split','trainorDev','Train/Dev','TrainOrTest']:
                    if cand in df_meta.columns:
                        split = str(meta_row.iloc[0][cand]).strip()
                        break
                if split is None:
                    # try common columns
                    for c in df_meta.columns:
                        if 'train' in c.lower() or 'dev' in c.lower() or 'split' in c.lower():
                            split = str(meta_row.iloc[0][c]).strip()
                            break
            # map binary label
            bin_label = None
            if diagnosis is not None:
                if diagnosis.upper() == 'HC':
                    bin_label = 0
                else:
                    # treat MCI and Dementia as 1
                    bin_label = 1
            rows.append({
                'participant_id': pid,
                'audio_path': ct_wav,
                'transcript_path': ct_txt,
                'diagnosis': diagnosis,
                'binary_label': bin_label,
                'split': split,
            })
    df_out = pd.DataFrame(rows)
    return df_out


def main():
    non_audio = find_non_audio_files()
    previews = preview_metadata_files()
    # pick metadata DF: prefer Data_AUG file
    meta_df = None
    for p in METADATA_CANDIDATES:
        if p.exists():
            try:
                meta_df = pd.read_csv(p)
                print(f"Using metadata file: {p}")
                break
            except Exception as e:
                print(f"Failed to read {p}: {e}")
    if meta_df is None:
        # try to find any csv in data_old
        for p in (ROOT / 'data_old').glob('*.csv'):
            try:
                meta_df = pd.read_csv(p)
                print(f"Using fallback metadata file: {p}")
                break
            except:
                continue
    if meta_df is None:
        print("No metadata CSV found; aborting.")
        return
    diag_col = identify_diagnosis_column(meta_df)
    print(f"Diagnosis column selected: {diag_col}")
    df_ctd = build_ctd_table(meta_df, diag_col)
    print("\n=== CTD table preview ===")
    print(df_ctd.head(20).to_string(index=False))
    # print class distribution
    counts = Counter(df_ctd['binary_label'].dropna())
    print("\nClass distribution:")
    print(counts)
    # train/test breakdown
    print("\nSplit breakdown:")
    print(df_ctd['split'].value_counts(dropna=False))
    # save preview
    df_ctd.to_csv(OUT_CSV, index=False)
    print(f"Saved CTD binary preview to {OUT_CSV}")

if __name__ == '__main__':
    main()
