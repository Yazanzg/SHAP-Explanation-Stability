# Attribution Stability vs Training-Set Size

**Thesis:** Are SHAP feature attributions in transcript-based cognitive-impairment
classifiers robust to which participants enter the training set, and how does this
robustness change as training size grows?

This repository contains the **Bachelor thesis code component** for PROCESS-2
Cookie Theft Description (CTD) transcripts. It extracts linguistic features from
transcripts, trains three model families with **fixed, pre-specified settings (no
tuning)**, computes SHAP attributions on a **fixed development split**, and
measures how the **stability of those attributions across different training
subsets** changes as the training-set size grows. The controlled PROCESS-2
dataset is **not** redistributed here.

> Terminology: "attribution stability" = similarity of SHAP attributions across
> models trained on different participant subsets of the same size;
> "training-size effect" = how that stability changes with subset size;
> "performance" = test-set AUROC / F1 / accuracy / precision / recall. The
> modelling object is **transcript-derived features**.

## What this code does

1. Loads a cohort master table (`outputs/process2_ctd_binary_master.csv`) with
   CTD transcript paths, the binary label (0 = HC, 1 = CI = MCI + dementia
   merged), and the predefined train/dev split.
2. Cleans transcript text and extracts **13 linguistic features** (spaCy).
3. Trains **Logistic Regression**, **Random Forest**, and **RBF SVM** on balanced,
   stratified subsets drawn from the 320-participant training pool, with fixed
   hyperparameters (no GridSearchCV, no performance-based selection).
4. Evaluates on the **fixed 80-participant development split** (AUROC, F1,
   accuracy, precision, recall).
5. Computes **SHAP** attributions on that same fixed split (LinearSHAP for LR,
   TreeSHAP for RF, KernelSHAP for SVM) and reduces each run to a mean-|SHAP|
   vector of length 13.
6. Measures **attribution stability** across many subsets of the same size
   (pairwise Pearson on raw mean-|SHAP| vectors as the primary metric; Jaccard@k
   and Spearman as secondary/appendix), with percentile intervals, as a function
   of training-set size.

## Configuration

All paths and parameters live in **`config.yaml`** at the repository root — the
single source of truth (paths, dataset schema, the 13 features, model settings,
training-size grid, iteration count, seeding scheme, Jaccard `k` values, SHAP
settings). The code is dataset-agnostic: point `dataset.master_csv` at any CSV
exposing the configured `participant_id` / `transcript_path` / `label` / `split`
columns. Set the `THESIS_CONFIG` environment variable to use an alternate config.

## Dataset (not included in this repository)

- **Source:** PROCESS-2 (controlled access via Hugging Face / CognoSpeak)
- **Task:** Cookie Theft Description (`*__CTD.txt`) only; HC vs CI (MCI + dementia)
- **Split:** predefined **train (320)** / **dev (80)**; never resampled
- Place data locally under `PROCESS-2/` (see `examples/` for the expected layout)

## Installation

```powershell
cd <repository-root>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-freeze.txt
python -m spacy download en_core_web_sm
```

## Smoke test (no real data required)

```powershell
$env:THESIS_CONFIG = "examples/synthetic/config.yaml"
python scripts/smoke_test.py
```

Runs the full pipeline on a 10-participant synthetic cohort in under two minutes.

## Reproduce (staged scripts)

Run from the repository root in order; see `scripts/README.md` for the full list.

```powershell
python scripts/validate_process2_loader.py
python scripts/validate_process2_preprocessing.py
python scripts/validate_process2_features.py
python scripts/audit_features.py
# training-size learning-curve runs, stability metrics, and plots: TASK 4-8
```

## Modelling features (n = 13)

`word_count`, `sentence_count`, `type_token_ratio`, `filler_count`,
`filler_ratio`, `mean_clause_length`, `content_density`, `noun_ratio`,
`verb_ratio`, `adjective_ratio`, `adverb_ratio`, `pronoun_ratio`,
`determiner_ratio`.

(`semantic_coherence` was constant 0.0 in the thesis environment and is excluded;
see `outputs/audit/feature_audit.md`.)

## Model settings

Models used fixed settings following scikit-learn defaults where possible; no
setting was selected through GridSearchCV, optimization, or performance-based
search. This is not hyperparameter tuning.

- **Logistic Regression** — scikit-learn defaults (`lbfgs`, `max_iter=100`,
  `C=1.0`); convergence verified (15-17 iterations on the smallest/largest
  subsets). Explained with LinearSHAP.
- **Random Forest** — scikit-learn defaults (`n_estimators=100`). Explained with
  TreeSHAP.
- **RBF SVM** — scikit-learn defaults (`C=1.0`, `gamma="scale"`). `probability=True`
  is an operational requirement for SHAP probability explanations (KernelSHAP
  reads `predict_proba`), not a performance setting. Explained with KernelSHAP
  (background = 24 rows, `nsamples=80`).

## Disclaimer

Research and education only. **Not** a clinical diagnosis system. Do not
redistribute PROCESS-2 audio or transcripts via this repository.

## Citation

If you use PROCESS-2, cite the dataset publication and Hugging Face access terms.
A full citations section (scikit-learn, SHAP, SciPy, NumPy, pandas, matplotlib,
spaCy) is finalised in the documentation pass (TASK 9).
