#!/usr/bin/env python3
"""Generate consolidated thesis pipeline notebooks (explained + standard).

Outputs:
  notebooks/thesis_pipeline_explained.ipynb  — student walkthrough (first person)
  notebooks/thesis_pipeline_standard.ipynb   — minimal headers + code only

Both cover the resubmission pipeline aligned with config.yaml, src/, and the
active scripts/ (training-size attribution stability; no GridSearchCV).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"
OUT_EXPLAINED = NOTEBOOKS_DIR / "thesis_pipeline_explained.ipynb"
OUT_STANDARD = NOTEBOOKS_DIR / "thesis_pipeline_standard.ipynb"


def cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def md(text: str) -> dict:
    return cell(text.rstrip() + "\n", "markdown")


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }


def setup_cells(explained: bool) -> list[dict]:
    if explained:
        return [
            md(
                """# Thesis Pipeline — Full Walkthrough (Explained)

**Student:** Yazan  
**Topic:** Attribution stability vs. training-set size in transcript-based cognitive-impairment classifiers

This notebook walks through the **resubmission pipeline**: I extract 13 linguistic features from PROCESS-2 Cookie Theft Description transcripts, train Logistic Regression / Random Forest / SVM-RBF on **balanced participant subsets** drawn from the 320-participant training pool (fixed hyperparameters, **no GridSearchCV**), evaluate and explain predictions on the **fixed 80-participant dev split**, and measure how **SHAP attribution stability** changes as training size grows.

## Pipeline steps (run cells top to bottom)

| Step | Section | What happens |
|------|---------|--------------|
| 1 | Setup | Verify project root, config, and key files |
| 2 | Data loading | Load master table (400 participants, train/dev split) |
| 3 | Preprocessing | Clean raw CTD text |
| 4 | Feature extraction | Build 13-feature matrix → `process2_features.csv` |
| 5 | Imputation audit | Confirm 0 NaNs; document why SimpleImputer was removed |
| 6 | Training-size experiment | Smoke demo or inspect frozen full results |
| 7 | Stability summaries | Pearson / Jaccard@k / performance vs. training size |
| 8 | Figures | Main panels N=80–280; N=320 boundary in tables only |

## Important design choices (for supervisor discussion)

- **No data leakage:** the dev split is never used for training, subset sampling, or SHAP background.
- **Fixed hyperparameters:** scikit-learn defaults where possible; nothing selected by performance search.
- **Deterministic subset order:** after sampling HC/CI participants, rows are sorted by `participant_id` before fitting (so attribution differences reflect *who* was sampled, not row order).
- **13 features:** `semantic_coherence` excluded (constant 0.0); see `outputs/audit/feature_audit.md`.
- **Authoritative scripts:** `scripts/` reproduce the thesis results; this notebook calls the same `src/` modules or inspects their outputs.

## Setup note

Run from the project root with `.venv` activated. PROCESS-2 transcripts stay local (not in the repo). Full LR+RF+SVM grid runs take ~15–30 minutes; for a quick demo use the smoke script or inspect pre-computed CSVs in `outputs/`."""
            ),
            cell(
                """# Run this setup cell first, then run the rest top to bottom
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

print("Project root:", ROOT)
print("Config:", config.CONFIG_PATH)
print("Master CSV exists:", config.MASTER_CSV.is_file())
print("Features CSV exists:", config.FEATURES_CSV.is_file())

expected = [
    "config.yaml",
    "src/config.py",
    "src/data_loading.py",
    "src/preprocessing.py",
    "src/feature_extraction.py",
    "src/modeling.py",
    "src/explainability.py",
    "scripts/validate_process2_loader.py",
    "scripts/validate_process2_preprocessing.py",
    "scripts/validate_process2_features.py",
    "scripts/audit_features.py",
    "scripts/run_training_size_stability.py",
    "scripts/compute_training_size_stability.py",
    "scripts/plot_training_size_stability.py",
]
for rel in expected:
    p = ROOT / rel
    print(("OK" if p.exists() else "MISSING"), rel)"""
            ),
        ]
    return [
        md("# Thesis Pipeline\n\nRun from the project root. Execute cells top to bottom.\n"),
        cell(
            """import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
print(ROOT)
print("config:", config.CONFIG_PATH)
print("master:", config.MASTER_CSV.is_file())
print("features:", config.FEATURES_CSV.is_file())"""
        ),
    ]


def step2_data_loading(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 2 — Data Loading (PROCESS-2 Master Table)\n"),
                md(
                    """## What this step does

I load the **master table** for PROCESS-2. Each row is one participant with:
- a local path to their CTD transcript (`.txt`)
- diagnosis (HC / MCI / Dementia) and binary label (0 = HC, 1 = CI)
- predefined split (**train = 320**, **dev = 80**)

I do **not** create a new random split — I use the official PROCESS-2 split so results stay comparable to the dataset documentation."""
                ),
            ]
        )
    else:
        cells.append(md("## 2. Data Loading\n"))

    if explained:
        code = """from src import config
from src.config import COL_LABEL, COL_PARTICIPANT_ID, COL_SPLIT, DIAGNOSIS_COLUMN
from src.data_loading import load_dataset_table

print("Master CSV:", config.MASTER_CSV)
table = load_dataset_table(load_text=True)

print(f"Rows: {len(table)}  |  Unique IDs: {table[COL_PARTICIPANT_ID].nunique()}")
if DIAGNOSIS_COLUMN in table.columns:
    print("\\nDiagnosis counts:\\n", table[DIAGNOSIS_COLUMN].value_counts())
print("\\nBinary counts (0=HC, 1=CI):\\n", table[COL_LABEL].value_counts().sort_index())
print("\\nSplit counts:\\n", table[COL_SPLIT].value_counts())

from pathlib import Path
missing = [pid for pid, p in zip(table[COL_PARTICIPANT_ID], table["transcript_path"])
           if not Path(p).is_file()]
print(f"Missing transcript files: {len(missing)}")

table.head(3)[[COL_PARTICIPANT_ID, COL_SPLIT, DIAGNOSIS_COLUMN, COL_LABEL]]"""
    else:
        code = """from src import config
from src.config import COL_LABEL, COL_PARTICIPANT_ID, COL_SPLIT, DIAGNOSIS_COLUMN
from src.data_loading import load_dataset_table
from pathlib import Path

table = load_dataset_table(load_text=True)
print(len(table), table[COL_PARTICIPANT_ID].nunique())
print(table[DIAGNOSIS_COLUMN].value_counts())
print(table[COL_LABEL].value_counts().sort_index())
print(table[COL_SPLIT].value_counts())
print("missing files:", sum(not Path(p).is_file() for p in table["transcript_path"]))
table.head(3)"""

    cells.append(cell(code))
    return cells


def step3_preprocessing(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 3 — Preprocessing (Transcript Cleaning)\n"),
                md(
                    """## What this step does

Raw CTD transcripts can contain timing markers like `(3 seconds)`, CHAT-style tags, and extra punctuation. Before feature extraction I clean text with `clean_transcript()` from `src/preprocessing.py`.

Important choices:
- I keep disfluencies (*um*, *uh*) because they are linguistically relevant
- I lowercase everything for consistent token counts
- CTD files are already participant-only speech"""
                ),
            ]
        )
    else:
        cells.append(md("## 3. Preprocessing\n"))

    if explained:
        code = """from src.data_loading import load_dataset_table
from src.preprocessing import clean_transcript

table = load_dataset_table(load_text=True)
table["cleaned_transcript"] = table["transcript"].astype(str).map(clean_transcript)

table["raw_length"] = table["transcript"].astype(str).str.len()
table["cleaned_length"] = table["cleaned_transcript"].str.len()
print("Raw length (chars):", table["raw_length"].describe()[["min", "50%", "max"]].to_dict())
print("Cleaned length:", table["cleaned_length"].describe()[["min", "50%", "max"]].to_dict())

sample = table.sample(1, random_state=42).iloc[0]
print("\\nParticipant:", sample["participant_id"])
print("RAW (first 300 chars):", sample["transcript"][:300])
print("\\nCLEANED (first 300 chars):", sample["cleaned_transcript"][:300])"""
    else:
        code = """from src.data_loading import load_dataset_table
from src.preprocessing import clean_transcript

table = load_dataset_table(load_text=True)
table["cleaned_transcript"] = table["transcript"].astype(str).map(clean_transcript)
print(table["transcript"].astype(str).str.len().describe())
print(table["cleaned_transcript"].str.len().describe())
sample = table.sample(1, random_state=42).iloc[0]
print(sample["participant_id"], sample["transcript"][:200], sample["cleaned_transcript"][:200], sep="\\n")"""

    cells.append(cell(code))
    return cells


def step4_features(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 4 — Feature Extraction (13 Linguistic Features)\n"),
                md(
                    """## What this step does

From each cleaned transcript I extract **13 numeric features** (spaCy + simple counts):
word/sentence counts, type-token ratio, fillers, mean clause length, content density, and POS ratios.

**`semantic_coherence` is excluded** — it was constant 0.0 in my environment. This matches `config.yaml` and `scripts/validate_process2_features.py`."""
                ),
            ]
        )
    else:
        cells.append(md("## 4. Feature Extraction\n"))

    if explained:
        code = """from src import config
from src.config import COL_LABEL, COL_PARTICIPANT_ID, COL_SPLIT, DIAGNOSIS_COLUMN, FEATURE_COLUMNS
from src.data_loading import load_dataset_table
from src.feature_extraction import build_feature_matrix

table = load_dataset_table(load_text=True)
feats = build_feature_matrix(
    table,
    text_column="transcript",
    id_column=COL_PARTICIPANT_ID,
    label_column=COL_LABEL,
    split_column=COL_SPLIT,
)
if DIAGNOSIS_COLUMN in table.columns:
    feats.insert(1, DIAGNOSIS_COLUMN, table[DIAGNOSIS_COLUMN].values)

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
feats.to_csv(config.FEATURES_CSV, index=False)

print("Saved:", config.FEATURES_CSV)
print("Shape:", feats.shape)
print(f"Modeling features ({len(FEATURE_COLUMNS)}):", list(FEATURE_COLUMNS))
print("Split counts:", feats[COL_SPLIT].value_counts().to_dict())
feats[[COL_PARTICIPANT_ID, COL_LABEL, COL_SPLIT, "word_count", "type_token_ratio", "filler_ratio"]].head()"""
    else:
        code = """from src import config
from src.config import COL_LABEL, COL_PARTICIPANT_ID, COL_SPLIT, DIAGNOSIS_COLUMN, FEATURE_COLUMNS
from src.data_loading import load_dataset_table
from src.feature_extraction import build_feature_matrix

table = load_dataset_table(load_text=True)
feats = build_feature_matrix(table, text_column="transcript",
    id_column=COL_PARTICIPANT_ID, label_column=COL_LABEL, split_column=COL_SPLIT)
if DIAGNOSIS_COLUMN in table.columns:
    feats.insert(1, DIAGNOSIS_COLUMN, table[DIAGNOSIS_COLUMN].values)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
feats.to_csv(config.FEATURES_CSV, index=False)
print(config.FEATURES_CSV, feats.shape)
feats.head(3)"""

    cells.append(cell(code))
    return cells


def step5_audit(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 5 — Imputation Audit & Feature Quality\n"),
                md(
                    """## What this step does

Before any modelling I audit the 13-feature matrix:
- count NaNs, infinities, and constant columns
- prove `SimpleImputer` is a no-op (0 NaNs → imputation removed from pipelines)
- save Pearson/Spearman correlation matrices for the discussion

Equivalent to: `python scripts/audit_features.py`"""
                ),
                cell(
                    """from src import config
import subprocess, sys
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "audit_features.py")],
    cwd=ROOT, capture_output=True, text=True,
)
print(result.stdout[-2500:] if len(result.stdout) > 2500 else result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("audit_features.py failed")

audit_md = config.AUDIT_DIR / "feature_audit.md"
print("\\nAudit report:", audit_md)
print(audit_md.read_text(encoding="utf-8")[:1200], "...")"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## 5. Imputation Audit\n"),
                cell(
                    """import subprocess, sys
subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_features.py")], cwd=ROOT, check=True)
(config.AUDIT_DIR / "feature_audit.md").read_text(encoding="utf-8")[:800]"""
                ),
            ]
        )
    return cells


def step6_training_size(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 6 — Training-Size Learning Curve (Attribution Stability)\n"),
                md(
                    """## What this step does

This is the **core resubmission experiment**. For each model and each training size in `[80, 120, 160, 200, 240, 280, 320]`:
1. Draw 20 **balanced** subsets (n_HC = n_CI = size/2) from the 320-participant train pool
2. Sort each subset by `participant_id` (deterministic order before fitting)
3. Fit LR / RF / SVM with **fixed hyperparameters**
4. Evaluate on the **fixed 80-participant dev split**
5. Compute SHAP on dev and reduce to a mean-|SHAP| vector (length 13)

**Smoke demo** (3 sizes × 5 repeats, LR only) runs in ~1 minute.  
**Full grid** (7 sizes × 20 repeats × 3 models = 420 fits) takes ~15–30 minutes.

For supervisor walkthrough I usually **inspect pre-computed outputs** in `outputs/` rather than re-run the full grid."""
                ),
                cell(
                    """from src import config
import pandas as pd
# Option A — quick smoke (LR only, 3 sizes, 5 repeats)
# Uncomment to run live:
# import subprocess, sys
# subprocess.run([sys.executable, str(ROOT / "scripts" / "run_training_size_stability.py"), "--smoke"],
#                cwd=ROOT, check=True)

# Option B — inspect frozen full results (recommended for walkthrough)
import pandas as pd
runs = pd.read_csv(config.OUTPUT_DIR / "training_size_runs.csv")
imp = pd.read_csv(config.IMPORTANCE_VECTORS_DIR / "training_size_importance_vectors.csv")
print("Run rows:", len(runs), " | importance vectors:", imp[["model","training_size","run_id"]].drop_duplicates().shape[0])
print("Models:", sorted(runs.model.unique()))
print("Training sizes:", sorted(runs.training_size.unique()))
runs.groupby(["model", "training_size"]).size().unstack(fill_value=0)"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## 6. Training-Size Experiment\n"),
                cell(
                    """import pandas as pd
runs = pd.read_csv(config.OUTPUT_DIR / "training_size_runs.csv")
imp = pd.read_csv(config.IMPORTANCE_VECTORS_DIR / "training_size_importance_vectors.csv")
print(len(runs), imp[["model","training_size","run_id"]].drop_duplicates().shape[0])
runs.groupby(["model", "training_size"]).size().unstack(fill_value=0)"""
                ),
            ]
        )
    return cells


def step7_summaries(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 7 — Stability & Performance Summaries\n"),
                md(
                    """## What this step does

From the per-run importance vectors I compute, for each (model, training size):
- **Pearson** (primary): mean pairwise correlation of mean-|SHAP| vectors across repeats
- **Jaccard@k** (k = 3, 5, 10): overlap of top-k features between repeat pairs
- **Spearman** (appendix)
- Dev **AUROC / F1 / accuracy** with percentile bands

Equivalent to: `python scripts/compute_training_size_stability.py --tag full`

**N=320** is the **full-pool boundary**: every repeat samples the same 320 participants, so attribution variation collapses (Pearson → 1.0 for LR/RF). Main figures use **N=80–280 only**."""
                ),
                cell(
                    """import pandas as pd
from IPython.display import display
import subprocess, sys
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "compute_training_size_stability.py"), "--tag", "full"],
    cwd=ROOT, capture_output=True, text=True,
)
print(result.stdout)

stab = pd.read_csv(config.OUTPUT_DIR / "training_size_stability_summary.csv")
perf = pd.read_csv(config.OUTPUT_DIR / "training_size_performance_summary.csv")

cols = ["model", "training_size", "pearson_mean", "pearson_lo", "pearson_hi",
        "jaccard5_mean", "jaccard5_lo", "jaccard5_hi"]
print("\\n=== Attribution stability (Pearson + Jaccard@5) ===")
display(stab[cols].round(3))

pcols = ["model", "training_size", "auroc_mean", "auroc_lo", "auroc_hi",
         "f1_mean", "f1_lo", "f1_hi"]
print("\\n=== Dev performance ===")
display(perf[pcols].round(3))"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## 7. Stability & Performance Summaries\n"),
                cell(
                    """import subprocess, sys, pandas as pd
subprocess.run([sys.executable, str(ROOT / "scripts" / "compute_training_size_stability.py"),
                "--tag", "full"], cwd=ROOT, check=True)
stab = pd.read_csv(config.OUTPUT_DIR / "training_size_stability_summary.csv")
perf = pd.read_csv(config.OUTPUT_DIR / "training_size_performance_summary.csv")
stab[["model","training_size","pearson_mean","pearson_lo","pearson_hi"]].round(3)
perf[["model","training_size","auroc_mean","auroc_lo","auroc_hi"]].round(3)"""
                ),
            ]
        )
    return cells


def step8_figures(explained: bool) -> list[dict]:
    cells: list[dict] = []
    if explained:
        cells.extend(
            [
                md("---\n\n# Step 8 — Figures (TASK 8)\n"),
                md(
                    """## What this step does

Generate the headline figures from the frozen sorted result set:
- **Fig 1:** Pearson attribution stability (N=80–280, 95% percentile error bars)
- **Fig 2:** Dev AUROC vs. training size
- **Fig 3:** Jaccard@5 vs. training size
- **Fig 4:** Per-model top-6 mean-|SHAP| trajectories

**N=320 is excluded** from all main panels (full-pool boundary). Captions are in `outputs/figures/figure_captions.md`.

Equivalent to: `python scripts/plot_training_size_stability.py`"""
                ),
                cell(
                    """from src import config
import subprocess, sys
from IPython.display import Image, display

result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "plot_training_size_stability.py")],
    cwd=ROOT, capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("plot script failed")

for name in [
    "fig1_attribution_stability_pearson.png",
    "fig2_performance_auroc.png",
    "fig3_attribution_stability_jaccard5.png",
]:
    p = config.FIGURES_DIR / name
    print("\\n", p.name)
    display(Image(filename=str(p)))"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## 8. Figures\n"),
                cell(
                    """import subprocess, sys
from IPython.display import Image, display
subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_training_size_stability.py")],
               cwd=ROOT, check=True)
display(Image(filename=str(config.FIGURES_DIR / "fig1_attribution_stability_pearson.png")))
display(Image(filename=str(config.FIGURES_DIR / "fig2_performance_auroc.png")))"""
                ),
            ]
        )
    return cells


def build_notebook(explained: bool) -> list[dict]:
    cells: list[dict] = []
    cells.extend(setup_cells(explained))
    for step_fn in (
        step2_data_loading,
        step3_preprocessing,
        step4_features,
        step5_audit,
        step6_training_size,
        step7_summaries,
        step8_figures,
    ):
        cells.extend(step_fn(explained))
    return cells


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")


def main() -> None:
    explained_cells = build_notebook(explained=True)
    standard_cells = build_notebook(explained=False)
    write_notebook(OUT_EXPLAINED, explained_cells)
    write_notebook(OUT_STANDARD, standard_cells)
    print(f"Wrote {OUT_EXPLAINED} ({len(explained_cells)} cells)")
    print(f"Wrote {OUT_STANDARD} ({len(standard_cells)} cells)")


if __name__ == "__main__":
    main()
