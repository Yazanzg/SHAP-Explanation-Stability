# TASK 8 figure captions

## fig1_attribution_stability_pearson

Figure 1. SHAP attribution stability versus training-set size for Logistic Regression, Random Forest, and SVM-RBF over N=80-280. Stability is the mean pairwise Pearson correlation between the per-repeat mean-|SHAP| feature-importance vectors (13 features), over all C(20,2)=190 repeat pairs per cell. Shaded bands / intervals are 95% percentile intervals (2.5th-97.5th percentile) computed across repeats. Higher values indicate more reproducible attributions across resampled training sets. Main figures use N=80-280; N=320 is excluded from all main figures because it is the full-pool boundary: at the full training pool every repeat draws the same participants, so participant-subset variation collapses and the metric is degenerate (reported in tables/appendix only).

## fig2_performance_auroc

Figure 2. Predictive performance (dev-set AUROC) versus training-set size for Logistic Regression, Random Forest, and SVM-RBF over N=80-280. Lines show the mean over 20 repeats per training size on the fixed 80-participant dev split. Shaded bands / intervals are 95% percentile intervals (2.5th-97.5th percentile) computed across repeats. Main figures use N=80-280; N=320 is excluded from all main figures because it is the full-pool boundary: at the full training pool every repeat draws the same participants, so participant-subset variation collapses and the metric is degenerate (reported in tables/appendix only).

## fig3_attribution_stability_jaccard5

Figure 3. Top-5 SHAP feature-set stability (Jaccard@5) versus training-set size for Logistic Regression, Random Forest, and SVM-RBF over N=80-280. Jaccard@5 is the mean overlap of the top-5 features (by mean-|SHAP|) between repeat pairs, over all C(20,2)=190 pairs per cell. Shaded bands / intervals are 95% percentile intervals (2.5th-97.5th percentile) computed across repeats. Main figures use N=80-280; N=320 is excluded from all main figures because it is the full-pool boundary: at the full training pool every repeat draws the same participants, so participant-subset variation collapses and the metric is degenerate (reported in tables/appendix only).

## fig4_feature_importance_trajectory_logistic_regression

Figure 4 (Logistic Regression). Mean-|SHAP| trajectories of the top 6 features (by average importance across N=80-280) for Logistic Regression. Each point is the mean absolute SHAP value averaged over the 20 repeats at that training size, on the fixed dev split. Shows how the model's most-attributed features evolve as the training set grows. Main figures use N=80-280; N=320 is excluded from all main figures because it is the full-pool boundary: at the full training pool every repeat draws the same participants, so participant-subset variation collapses and the metric is degenerate (reported in tables/appendix only).

## fig4_feature_importance_trajectory_random_forest

Figure 4 (Random Forest). Mean-|SHAP| trajectories of the top 6 features (by average importance across N=80-280) for Random Forest. Each point is the mean absolute SHAP value averaged over the 20 repeats at that training size, on the fixed dev split. Shows how the model's most-attributed features evolve as the training set grows. Main figures use N=80-280; N=320 is excluded from all main figures because it is the full-pool boundary: at the full training pool every repeat draws the same participants, so participant-subset variation collapses and the metric is degenerate (reported in tables/appendix only).

## fig4_feature_importance_trajectory_svm_rbf

Figure 4 (SVM-RBF). Mean-|SHAP| trajectories of the top 6 features (by average importance across N=80-280) for SVM-RBF. Each point is the mean absolute SHAP value averaged over the 20 repeats at that training size, on the fixed dev split. Shows how the model's most-attributed features evolve as the training set grows. Main figures use N=80-280; N=320 is excluded from all main figures because it is the full-pool boundary: at the full training pool every repeat draws the same participants, so participant-subset variation collapses and the metric is degenerate (reported in tables/appendix only).

