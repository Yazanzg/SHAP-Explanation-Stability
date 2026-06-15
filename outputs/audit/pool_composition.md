# Pool composition verification (TASK 3)

Source: `outputs/process2_ctd_binary_master.csv` (split values normalised; `test` → `dev`).

## Training pool (the 320 predefined training participants)

| Class | Count |
|---|---|
| HC (label 0) | **160** |
| CI (label 1) | **160** |
| **Total** | **320** |

CI breakdown: MCI 120 + Dementia 40.

## Fixed test set (the 80 predefined dev participants)

| Class | Count |
|---|---|
| HC (label 0) | 40 |
| CI (label 1) | 40 |
| **Total** | 80 |

CI breakdown: MCI 30 + Dementia 10.

## Size-grid ceiling

- Maximum balanced subset size = 2 × min(HC, CI) = 2 × min(160, 160) = **320**.
- The pool is **exactly symmetric (160/160)**, so the training-size grid runs to
  the full **320** with no adjustment.
- Configured grid (`config.yaml` → `experiment.training_size_grid`):
  `[80, 120, 160, 200, 240, 280, 320]` (7 sizes). Each subset is balanced
  (n_HC = n_CI = size / 2), sampled independently per class without replacement.

**Awaiting confirmation of the grid before TASK 4 launches.**
