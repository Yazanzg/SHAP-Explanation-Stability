# Scripts

Run from the repository root after installing dependencies. The pipeline is
config-driven: `THESIS_CONFIG` selects which `config.yaml` to load (defaults to
the repository-root `config.yaml`).

| Stage | Script | Purpose / main outputs |
|---|---|---|
| Smoke test | `smoke_test.py` | Runs the full pipeline on the 10-row synthetic example (`examples/synthetic/`) in < 2 min |
| Data load check | `validate_process2_loader.py` | Validates the cohort master table loads correctly |
| Preprocessing check | `validate_process2_preprocessing.py` | Validates transcript cleaning |
| Feature extraction | `validate_process2_features.py` | Builds the 13-feature matrix (`outputs/process2_features.csv`) + validation report |
| Imputation audit | `audit_features.py` | `outputs/audit/feature_audit.md`, `feature_correlations.csv/.png` |
| Learning-curve runs | `run_training_size_stability.py` | (TASK 4+) per-subset performance + SHAP importance vectors |
| Stability metrics | `compute_training_size_stability.py` | (TASK 5) Pearson / Jaccard@k / Spearman summaries |
| Plots | `plot_training_size_stability.py` | (TASK 8) headline figures |

Feature-extraction asserts split sizes (train=320, dev=80) and the 13-feature
contract. The training-size scripts implement the attribution-stability design
(stability of SHAP attributions across participant subsets as training size grows).
