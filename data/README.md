# Data Directory

This directory contains the TCGA datasets used for benchmarking.

## Sources
Data is sourced from **TCGA (The Cancer Genome Atlas)** via Genomic Data Commons (GDC).

## Subdirectories
- `TCGA-BRCA/`: Breast Invasive Carcinoma.
- `TCGA-COAD/`: Colon Adenocarcinoma.
- `TCGA-PRAD/`: Prostate Adenocarcinoma.

## Files
Each subdirectory contains:
- `*_expression_matrix.csv`: Log-transformed or normalized gene expression values. Rows are genes, columns are samples (or vice versa, depending on the script loading logic).
- `*_subtypes.csv`: Clinical metadata including the "Subtype_Selected" column used as the target for classification.

## Notes on Preprocessing
The raw TCGA sample IDs (e.g., `TCGA-XX-XXXX-01A-...`) are truncated to the first 12 characters (e.g., `TCGA-XX-XXXX`) to match clinical records. Duplicate samples for the same patient are averaged during loading.
