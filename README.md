# Stability in Optimized NLP Models

**Thesis:** Evaluating Explanation Stability in Optimized NLP Models for Cognitive Impairment Classification from Spontaneous Speech

This repository contains the **Bachelor thesis code component** for PROCESS-2 Cookie Theft Description (CTD) transcripts. It provides reproducible scripts, validation reports, and reported results. The controlled PROCESS-2 dataset is **not** redistributed here.

## What this code does

1. Loads a local master table (`outputs/process2_ctd_binary_master.csv`) with CTD transcript paths and labels
2. Preprocesses transcript text and extracts **13 linguistic features** (spaCy; semantic coherence excluded in final thesis runs)
3. Trains **Logistic Regression**, **Random Forest**, and **RBF SVM** on the predefined **training split** (320 participants)
4. Tunes hyperparameters with **GridSearchCV** on training data only (`roc_auc`, 5-fold stratified CV)
5. Evaluates on the predefined **development split** (80 participants)
6. Computes **SHAP** explanations on the **fixed development split**
7. Measures **explanation stability** (direct baseline vs optimized SHAP; bootstrap resampling of training data)

## Dataset (not included in this repository)

- **Source:** PROCESS-2 (controlled access via Hugging Face / CognoSpeak)
- **Task:** Cookie Theft Description (`*__CTD.txt`) only
- **Labels:** `binary_label` — 0 = HC, 1 = CI (MCI + Dementia merged)
- **Split:** predefined **train** (320) / **dev** (80); no new random split is created locally
- **Place data locally** under `PROCESS-2/` (see `examples/` for expected file layout)

After downloading PROCESS-2, build or update the master table paths, then run the staged scripts below.

## Installation

```powershell
cd <repository-root>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-freeze.txt
python -m spacy download en_core_web_sm
```

Pinned versions used for thesis experiments are listed in `requirements-freeze.txt` (Python 3.11+ recommended).

## Reproduce thesis outputs (staged scripts)

Run from the repository root in order. Each step writes CSV/report files under `outputs/`.

```powershell
python scripts/validate_process2_loader.py
python scripts/validate_process2_preprocessing.py
python scripts/validate_process2_features.py
python scripts/validate_process2_baselines.py
python scripts/validate_process2_optimization.py
python scripts/validate_process2_shap.py
python scripts/validate_process2_stability.py
python scripts/validate_process2_bootstrap_stability.py
python scripts/plot_process2_stability_comparison.py
```

**Notes:**

- `validate_process2_features.py` requires `outputs/process2_ctd_binary_master.csv` with valid local `transcript_path` entries.
- **Semantic coherence** is computed in code but was **constant (0.0)** in the thesis environment and **excluded** from all modeling and SHAP (13 features only).
- **Bootstrap iterations:** 50 for Logistic Regression and Random Forest; **25 for SVM RBF** (runtime; documented in validation report).
- `run_pipeline.py` is a legacy/monolithic entry point; the **staged `scripts/validate_process2_*.py` files** match the reported thesis experiments.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `THESIS_DATASET` | `PROCESS2` | Dataset mode (`PROCESS2` for thesis) |
| `THESIS_DATA_DIR` | `<repo>/PROCESS-2` | PROCESS-2 root folder |
| `THESIS_PROCESS2_MASTER_CSV` | `outputs/process2_ctd_binary_master.csv` | Master index CSV |
| `THESIS_RANDOM_SEED` | `42` | Random seed |
| `THESIS_SHAP_BG` | `24` | KernelSHAP background size (from train) |
| `THESIS_KERNEL_NSAMPLES` | `80` | KernelSHAP `nsamples` |
| `THESIS_SPACY_MODEL` | `en_core_web_sm` | spaCy model |

## Key outputs (thesis-reported)

| File | Description |
|------|-------------|
| `outputs/process2_features.csv` | 400 rows, 13 modeling features + metadata |
| `outputs/baseline_results.csv` | Dev metrics: LR, RF, SVM baselines |
| `outputs/optimized_results.csv` | Dev metrics after GridSearchCV |
| `outputs/shap/process2_shap_values_{baseline\|optimized}_*.csv` | SHAP on **dev** (80 rows each) |
| `outputs/direct_shap_stability.csv` | Direct baseline vs optimized stability |
| `outputs/bootstrap_stability.csv` | Bootstrap stability summary |
| `outputs/figures/process2_stability_direct_vs_bootstrap_adjusted.png` | Results figure |

Validation `.txt` reports are written locally when you run staged scripts but are not stored in this repository.

## Project layout

```text
README.md
requirements.txt
requirements-freeze.txt
run_pipeline.py
src/                    # pipeline modules
scripts/                # staged PROCESS-2 validation + figure script
examples/               # dummy schema + sample transcript (no real data)
outputs/                # generated thesis artifacts (subset committed)
.gitignore
```

## Modeling features (final thesis, n=13)

`word_count`, `sentence_count`, `type_token_ratio`, `filler_count`, `filler_ratio`, `mean_clause_length`, `content_density`, `noun_ratio`, `verb_ratio`, `adjective_ratio`, `adverb_ratio`, `pronoun_ratio`, `determiner_ratio`

## Explanation stability (thesis focus)

1. **Direct SHAP stability** — Compare baseline vs optimized global feature importance (mean \|SHAP\|) on the **fixed development split**. Metrics: Spearman rank correlation, Jaccard@k, mean CV of \|SHAP\| across dev participants.
2. **Bootstrap stability** — Resample **training** data (320 with replacement), refit models with **fixed** hyperparameters, recompute SHAP on the **same development split**. Aggregate pairwise Spearman and Jaccard@k across bootstrap runs.

## Disclaimer

Research and education only. **Not** a clinical diagnosis system. Do not redistribute PROCESS-2 audio or transcripts via this repository.

## Citation

If you use PROCESS-2, cite the dataset publication and Hugging Face access terms. This code repository documents the thesis implementation only.
