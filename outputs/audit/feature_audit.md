# Feature audit — imputation and collinearity (TASK 2)

- Feature matrix: `C:\Users\Yazan\Thesis\outputs\process2_features.csv`
- Rows: 400  |  Configured features: 13
- Total NaNs across the 13 features: **0**
- Total infinite values: **0**
- Constant features: **none**

## Per-feature audit

| feature | dtype | nan | inf | constant | n_unique | min | max |
|---|---|---|---|---|---|---|---|
| word_count | float64 | 0 | 0 | False | 214 | 12 | 772 |
| sentence_count | float64 | 0 | 0 | False | 23 | 1 | 25 |
| type_token_ratio | float64 | 0 | 0 | False | 365 | 0.3627 | 1 |
| filler_count | float64 | 0 | 0 | False | 31 | 0 | 38 |
| filler_ratio | float64 | 0 | 0 | False | 306 | 0 | 0.1429 |
| mean_clause_length | float64 | 0 | 0 | False | 360 | 3.232 | 32 |
| content_density | float64 | 0 | 0 | False | 333 | 0.3208 | 0.5625 |
| noun_ratio | float64 | 0 | 0 | False | 336 | 0.09836 | 0.4375 |
| verb_ratio | float64 | 0 | 0 | False | 309 | 0.0625 | 0.2308 |
| adjective_ratio | float64 | 0 | 0 | False | 297 | 0 | 0.1068 |
| adverb_ratio | float64 | 0 | 0 | False | 294 | 0 | 0.2093 |
| pronoun_ratio | float64 | 0 | 0 | False | 328 | 0 | 0.2308 |
| determiner_ratio | float64 | 0 | 0 | False | 325 | 0.01639 | 0.2366 |

## NaN cross-tabulation by class (HC vs CI)

| feature | nan_HC | nan_CI |
|---|---|---|
| word_count | 0 | 0 |
| sentence_count | 0 | 0 |
| type_token_ratio | 0 | 0 |
| filler_count | 0 | 0 |
| filler_ratio | 0 | 0 |
| mean_clause_length | 0 | 0 |
| content_density | 0 | 0 |
| noun_ratio | 0 | 0 |
| verb_ratio | 0 | 0 |
| adjective_ratio | 0 | 0 |
| adverb_ratio | 0 | 0 |
| pronoun_ratio | 0 | 0 |
| determiner_ratio | 0 | 0 |

## Imputation decision

The feature matrix contains **zero missing values and zero infinities** across all configured features, and missingness does not exist in either class. The median `SimpleImputer` is therefore an exact no-op: applying it leaves the matrix unchanged (`matrix_unchanged=True`) and model predictions on the fixed dev split are bit-identical with and without it (`predictions_identical=True`). The imputer has been removed from all active model pipelines. No imputation is performed anywhere in the pipeline; because hand-crafted linguistic features cannot be randomly missing, imputation under a missing-at-random assumption would be statistically inappropriate, and there is nothing to impute in any case.

## Feature collinearity (for the discussion)

Pearson and Spearman 13x13 matrices saved to `feature_correlations.csv` and `feature_correlations.png` (+ PDF). Highly correlated pairs (|Pearson| >= 0.7):

| feature_a | feature_b | pearson | spearman |
|---|---|---|---|
| type_token_ratio | word_count | -0.818 | -0.887 |
| filler_count | filler_ratio | 0.729 | 0.840 |
| sentence_count | word_count | 0.713 | 0.720 |

## Note: imputation removed from the active pipeline

The first-submission scripts that used `SimpleImputer` (GridSearchCV optimisation, the old baseline/optimised SHAP and bootstrap-stability validators, and the monolithic `run_pipeline.py`) have been removed from the active branch as part of the training-size reframing. The only model construction path in the resubmission is `src/modeling.build_estimator`, which is imputer-free.
