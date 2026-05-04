# Wrapper-Based Feature Selection

This directory will contain the implementation of wrapper-based feature selection methods.

## Planned Methods
- **Recursive Feature Elimination (RFE):** Recursively removes features by importance until a target number of features is reached.
- **Sequential Feature Selection (SFS):** Greedy search that adds (Forward) or removes (Backward) features based on model performance.

## Expected Workflow
1. Use a base estimator (e.g., Logistic Regression) to evaluate feature subsets.
2. Search for the optimal subset using the wrapper strategy.
3. Evaluate final performance and compare with baseline and other methods.
