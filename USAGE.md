# Usage Guide

This document provides step-by-step instructions on how to set up and run the BRCA Classification benchmarking project.

## 1. Prerequisites

Ensure you have Python 3.8 or higher installed. It is recommended to use a virtual environment.

### Clone the Repository
```bash
git clone <repository-url>
cd 4824BRCAClassification
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 2. Data Preparation

The scripts expect the TCGA data to be present in the `data/` directory.

The directory structure should look like this:
```
data/
├── TCGA-BRCA/
│   ├── brca_expression_matrix.csv
│   └── brca_subtypes.csv
├── TCGA-COAD/
│   ├── coad_expression_matrix.csv
│   └── coad_subtypes.csv
└── TCGA-PRAD/
    ├── prad_expression_matrix.csv
    └── prad_subtypes.csv
```

## 3. Running the Experiments

All scripts should be executed from the **root directory** of the project.

### Running Baseline Models
The baseline scripts train a Logistic Regression model on the full feature set for each cancer type.
```bash
# Run individual baselines
python Baseline/brca_baseline.py
python Baseline/coad_baseline.py
python Baseline/prad_baseline.py
```

### Running Embedded Method (LASSO)
The LASSO script automatically iterates through all three datasets and performs feature selection.
```bash
python Embedded/run_lasso.py
```

### Running Filter and Wrapper Methods (Planned)
Instructions will be updated as these modules are implemented.
```bash
# Planned
python Filter/run_filter.py
python Wrapper/run_wrapper.py
```

## 4. Viewing Results

Results are organized by method in their respective `results/` directories.

### Baseline Results
Located in `Baseline/results/`:
- Check `*_results.txt` for numerical performance.
- View `*_confusion_matrix.png` for prediction heatmaps.

### Embedded Results
Located in `Embedded/results/LASSO/`:
- Check `lasso_summary.csv` for a cross-dataset comparison.
- Each cancer type has its own folder with detailed lists of selected features.

## 5. Troubleshooting

- **ModuleNotFoundError:** Ensure you are running the scripts from the root directory and that your `PYTHONPATH` includes the current directory.
- **FileNotFoundError:** Verify that the `data/` directory is populated and follows the naming convention outlined in `data/README.md`.
- **Absolute Paths:** If you encounter errors related to `H:/MLFinalProj/...`, ensure you have the latest version of the scripts which use relative paths.
