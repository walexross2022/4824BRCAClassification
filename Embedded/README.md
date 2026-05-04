# Embedded Feature Selection (LASSO)

This directory implements feature selection using the **LASSO (L1 regularization)** method.

## Concept
Embedded methods perform feature selection during the model training process. LASSO adds a penalty proportional to the absolute value of the coefficients, which forces some coefficients to exactly zero, effectively selecting a subset of features.

## Implementation
- **Algorithm:** Logistic Regression with `penalty='l1'` and `solver='liblinear'`.
- **Hyperparameter:** Regularization strength `C=0.1` (lower C means stronger regularization/fewer features).
- **Automation:** `run_lasso.py` automatically processes all three datasets (BRCA, COAD, PRAD).

## Files
- `run_lasso.py`: Main execution script that iterates through datasets and saves results.
- `feature_selection_lasso.py`: (Internal) Logic for LASSO selection.
- `model_baseline.py`: (Internal) Reusable model components.

## Results
Results are saved in `Embedded/results/LASSO/`:
- `*_lasso_results.txt`: Performance metrics and a list of selected features.
- `lasso_summary.csv`: Aggregated results for all datasets.
- Visualizations for each dataset.
