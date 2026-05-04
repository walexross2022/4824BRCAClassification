# Baseline Models (Control Group)

This directory contains scripts to establish a performance baseline for the three cancer datasets without any feature selection.

## Workflow
1. **Data Loading:** Merges gene expression matrices with subtype labels.
2. **Preprocessing:** 
   - Sample IDs are truncated to 12 characters to match clinical data.
   - Duplicates are averaged.
   - Data is scaled using `StandardScaler`.
3. **Training:** Logistic Regression with `max_iter=1000`.
4. **Evaluation:** Calculates Accuracy, Macro F1, and Weighted F1.

## Files
- `brca_baseline.py`: Baseline for Breast Cancer.
- `coad_baseline.py`: Baseline for Colon Cancer.
- `prad_baseline.py`: Baseline for Prostate Cancer.
- `check_data.py`: Utility to verify data alignment.

## Results
Results are saved in `Baseline/results/`:
- `*_results.txt`: Summary of metrics and classification report.
- `*_confusion_matrix.png`: Visual heatmap of model predictions.
- `*_metrics_bar.png`: Performance overview chart.
