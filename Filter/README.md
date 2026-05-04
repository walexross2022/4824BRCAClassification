# Filter-Based Feature Selection

This directory will contain the implementation of filter-based feature selection methods.

## Planned Methods
- **Variance Threshold:** Remove features with low variance.
- **ANOVA F-test:** Select features based on their relationship with the target variable (linear).
- **Mutual Information:** Capture non-linear relationships between features and the target.

## Expected Workflow
1. Apply filter method to the training set.
2. Select top `k` features (or features above a threshold).
3. Train a Logistic Regression model on the reduced feature set.
4. Evaluate and compare with baseline.
