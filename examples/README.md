# Example data (no real participants)

These files show the **expected column names and file layout** for running the pipeline without uploading PROCESS-2.

| File | Purpose |
|------|---------|
| `example_master_table_schema.csv` | Column schema for `process2_ctd_binary_master.csv` |
| `dummy_ctd.txt` | Minimal Cookie Theft Description text for path checks |
| `synthetic/` | Self-contained 10-participant fake cohort + config for the smoke test |

To run the full thesis pipeline you need the real PROCESS-2 cohort (400 participants) under `PROCESS-2/` with local paths in your master CSV. The committed `outputs/process2_ctd_binary_master.csv` documents the thesis run but points to paths on the author's machine; rebuild paths after cloning.

## Smoke test (no real data required)

`examples/synthetic/` contains 10 fake participants (distinct Cookie-Theft-style
transcripts), a `master.csv`, and a complete `config.yaml`. The smoke test runs
the full pipeline — load, clean, extract the 13 features, fit all three model
families, compute SHAP — end-to-end in well under two minutes:

```powershell
$env:THESIS_CONFIG = "examples/synthetic/config.yaml"
python scripts/smoke_test.py
```

The pipeline is config-driven: `THESIS_CONFIG` selects which `config.yaml` to
load (defaults to the repository-root `config.yaml`). Outputs land under
`examples/synthetic/outputs/` (git-ignored).

```powershell
# Example: test preprocessing on dummy text only (no models)
python -c "from src.preprocessing import clean_transcript; print(clean_transcript(open('examples/dummy_ctd.txt').read()))"
```
