# Thesis NLP Pipeline: PROCESS-2, SHAP, and Explanation Stability

This repository supports the thesis **“Evaluating Explanation Stability in Optimized NLP Models for Cognitive Impairment Classification from Spontaneous Speech”**.

The **final thesis dataset** is **PROCESS-2** (controlled access via Hugging Face / CognoSpeak). The pipeline:

1. Loads **Cookie Theft Description (CTD)** transcripts and `metadata.csv`
2. Maps **HC vs cognitive impairment (CI)** where CI = MCI + Dementia
3. Uses the **predefined Train/Test split** from metadata (no random test split)
4. Extracts linguistic features (spaCy + SBERT)
5. Trains **Logistic Regression**, **Random Forest**, and **RBF SVM** baselines
6. Runs **GridSearchCV** on the training set only
7. Evaluates held-out **Test** performance
8. Computes **SHAP** explanations and **explanation stability** metrics
9. Writes CSVs and figures under `outputs/`

## Important disclaimer

This code is for **research and education only**. It is **not** a clinical diagnosis system.

## Dataset: PROCESS-2

### Location

Default path:

```text
C:\Users\Yazan\Thesis\PROCESS-2\
```

Override with:

```powershell
$env:THESIS_DATA_DIR = "C:\path\to\PROCESS-2"
$env:THESIS_DATASET = "PROCESS2"
```

### Expected layout

```text
PROCESS-2/
├── metadata.csv
├── PROCESS-2_rec__001/
│   ├── PROCESS-2_rec__001__CTD.txt
│   ├── PROCESS-2_rec__001__SFT.txt   (not used in primary pipeline)
│   └── ...
└── PROCESS-2_rec__002/
    └── ...
```

### metadata.csv

Required columns:

| Column | Description |
|--------|-------------|
| Participant ID column | e.g. `participant_id` matching folder names (`PROCESS-2_rec__001`) |
| `diagnosis` | `HC`, `MCI`, or `Dementia` |
| `Split` | `Train` or `Test` |

### Binary label mapping (thesis target)

| Original | `diagnosis_label` |
|----------|-------------------|
| HC | 0 |
| MCI | 1 |
| Dementia | 1 |

Expected balance: **200 HC / 200 CI** (warning logged if counts differ).

### Task scope

- **Primary:** CTD only (`*__CTD.txt`)
- **Not used in this pipeline:** SFT, PFT (files may exist but are ignored)

### Splits

- **Train:** cross-validation, hyperparameter tuning, shuffled-label sanity check, SHAP stability
- **Test:** final held-out evaluation only (never used for tuning or SHAP background fitting)

## Legacy development mode

To use the previous `Data/` or `data/` layout (Archive, `Data_AUG_*.csv`):

```powershell
$env:THESIS_DATASET = "LEGACY"
```

All samples are treated as training data (no held-out test). See `src/data_loading.py`.

## Installation

```powershell
cd C:\Users\Yazan\Thesis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the full pipeline

```powershell
cd C:\Users\Yazan\Thesis
.\.venv\Scripts\Activate.ps1
python run_pipeline.py
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `THESIS_DATASET` | `PROCESS2` | `PROCESS2` or `LEGACY` |
| `THESIS_DATA_DIR` | `PROJECT_ROOT/PROCESS-2` | Dataset root |
| `THESIS_INCLUDE_BALANCED` | `0` | `1` adds `class_weight='balanced'` model variants |
| `THESIS_RANDOM_SEED` | `42` | Reproducibility |
| `THESIS_SKIP_SVM_STABILITY` | `1` | Skip SVM in CV/bootstrap stability (faster) |
| `THESIS_SHAP_BG` | `24` | SHAP background size (train only) |
| `THESIS_KERNEL_NSAMPLES` | `80` | KernelSHAP samples |
| `THESIS_BOOTSTRAPS` | `15` | Bootstrap stability runs |
| `THESIS_EXPECTED_HC` / `THESIS_EXPECTED_CI` | `200` / `200` | Balance warnings |

## Outputs (`outputs/`)

| File | Description |
|------|-------------|
| `features.csv` | `participant_id`, `diagnosis`, `diagnosis_label`, `split`, transcript, features |
| `baseline_results.csv` | Metrics with `eval_scope`: `train_cv` and `test` |
| `optimized_results.csv` | Tuned models; train CV + test |
| `shuffled_label_results.csv` | Shuffled labels on **train** only (`shuffled_train_cv`) |
| `shap_values_baseline.csv` | SHAP on **training** set |
| `shap_values_optimized.csv` | SHAP on **training** set |
| `stability_results.csv` | Spearman, Jaccard top-5, CV of \|SHAP\| (train only) |
| `figures/*.png` | Performance, stability, trade-off plots |

## Supervisor notebook

```powershell
jupyter notebook notebooks\supervisor_code_walkthrough.ipynb
```

Walkthrough of architecture, PROCESS-2 inspection, features, results, SHAP, and stability (reads `outputs/`; does not run the full pipeline by default).

## Project layout

- `src/` — modules (`config`, `data_loading`, `preprocessing`, `feature_extraction`, `modeling`, `optimization`, `explainability`, `stability`, `visualization`)
- `run_pipeline.py` — CLI entry point
- `PROCESS-2/` — place the final dataset here
- `outputs/` — generated artifacts

## Explanation stability (thesis focus)

Optimization may improve predictive performance, but explanations must remain **stable** and **interpretable**. Stability is measured on the **training set** via:

1. **5-fold stratified CV** — SHAP on each training fold; compare global \|SHAP\| rankings
2. **Bootstrap resamples** of the training set
3. **Baseline vs optimized** global SHAP profile agreement

Metrics: Spearman rank correlation, Jaccard(top-5 features), mean CV of \|SHAP\| magnitudes.
