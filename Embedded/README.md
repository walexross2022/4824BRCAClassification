# Embedded Feature Selection (LASSO & Random Forest)

This directory implements embedded feature selection methods following the **Professor's Example Workflow**.

## Workflow Standards
To ensure rigorous benchmarking, both methods adhere to the following steps:
1.  **Data Partitioning:** 80% Training / 20% Test (Stratified).
2.  **Normalization:** `StandardScaler` fitted on training data only.
3.  **Hyperparameter Tuning:** K-fold Cross-Validation on the training set.
4.  **Metrics:** Accuracy, Macro F1-score, Weighted ROC-AUC, Precision, and Recall.

## Methods

### 1. LASSO (L1 Regularization)
- **Script:** `run_lasso.py`
- **Logic:** Uses `LogisticRegressionCV` to find the optimal regularization strength `C` while performing feature selection (forcing irrelevant coefficients to zero).
- **Selection:** Features with non-zero coefficients.

### 2. Random Forest (Importance Ranking)
- **Script:** `run_rf.py`
- **Logic:** Uses `GridSearchCV` to tune `n_estimators` and `max_depth`.
- **Selection:** Features with `feature_importances_` greater than the mean importance across all features.

## Files
- `data_utils.py`: Shared utility for standardized TCGA data loading and alignment.
- `run_lasso.py`: Execution script for LASSO benchmarking.
- `run_rf.py`: Execution script for Random Forest benchmarking.
- `archive/`: Legacy scripts moved for project cleanliness.

## Results
Results are saved in `Embedded/results/`:
- `LASSO/`: Reports and confusion matrices for LASSO runs.
- `RandomForest/`: Reports and confusion matrices for RF runs.
- `summary_metrics.csv`: Aggregated performance metrics for all datasets.
