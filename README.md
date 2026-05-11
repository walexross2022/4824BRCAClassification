README
TCGA Cancer Subtype Classification Using Machine Learning

This project investigates the use of Machine Learning methods for multiclass cancer subtype classification using high-dimensional gene expression data from The Cancer Genome Atlas (TCGA). The goal is to compare baseline, filter-based, embedded, and wrapper-style feature selection approaches for reducing dimensionality while maintaining strong subtype classification performance.

The project evaluates multiple machine learning pipelines across three TCGA cancer datasets:

BRCA — Breast Invasive Carcinoma
COAD — Colon Adenocarcinoma
PRAD — Prostate Adenocarcinoma

The datasets contain approximately 60,660 gene expression features per sample, making dimensionality reduction and feature selection critical components of the workflow.

Project Goals

The primary objectives of this project are:

Perform multiclass cancer subtype classification using gene expression data
Compare different feature selection strategies
Evaluate dimensionality reduction effectiveness
Analyze tradeoffs between runtime, interpretability, and predictive performance
Investigate how aggressive feature reduction affects classification quality
Datasets

The project uses TCGA gene expression matrices and associated subtype labels.

Included Cancer Types
Dataset	Description
BRCA	Breast cancer molecular subtypes
COAD	Colon cancer molecular subtypes
PRAD	Prostate cancer subtypes

Each dataset includes:

Gene expression matrix
Cancer subtype labels
Patient/sample alignment preprocessing
Project Structure
4824BRCAClassification/
│
├── Baseline/
│   ├── baseline.py
│   └── results/
│
├── Filter/
│   ├── app.py
│   ├── variance_filter.py
│   ├── anova_filter.py
│   ├── mutual_information_filter.py
│   ├── run_filters.py
│   └── results/
│
├── Embedded/
│   ├── lasso_feature_selection.py
│   ├── generate_comparison_plots.py
│   └── results/
│
├── Wrapper/
│   └── (under development)
│
├── data/
│   ├── TCGA-BRCA/
│   ├── TCGA-COAD/
│   └── TCGA-PRAD/
│
└── README.md
Methodology
1. Baseline Models

Baseline models train directly on the full gene expression matrix with minimal feature reduction. These experiments establish reference performance metrics for comparison against feature selection approaches.

Metrics evaluated include:

Accuracy
Macro F1 Score
Weighted F1 Score
ROC-AUC
2. Filter Methods

Filter methods evaluate features independently of the classifier and provide fast, interpretable dimensionality reduction.

Implemented Filter Methods
Variance Threshold

Removes genes with very low variance across samples.

Threshold sweep tested:

0.05, 0.10, 0.25, 0.50, 0.75

This method serves as a lightweight runtime-efficient baseline.

ANOVA / F-Test

Ranks genes using ANOVA F-statistics and selects the top K features.

Mutual Information

Ranks genes using mutual information scores to capture nonlinear relationships between genes and cancer subtypes.

3. Embedded Methods

Embedded methods perform feature selection during model training.

Implemented Embedded Methods
LASSO Logistic Regression

Uses L1 regularization to encourage sparse feature selection and automatically remove less important genes.

The project evaluates:

Number of selected genes
Classification performance
Runtime tradeoffs
Feature reduction effectiveness
4. Wrapper Methods

Wrapper-based methods are planned for future implementation.

Potential methods include:

Recursive Feature Elimination (RFE)
Forward Selection
Backward Elimination

Due to the extremely high dimensionality of TCGA datasets, wrapper methods are expected to be computationally expensive.

Preprocessing Pipeline

The preprocessing workflow includes:

Loading expression matrices
Loading subtype labels
Standardizing TCGA sample IDs
Removing duplicate patient samples
Aligning labels with expression data
Train/test splitting
Feature scaling using StandardScaler

All preprocessing is centralized through:

Filter/app.py

to ensure consistent dataset handling across experiments.

Evaluation Metrics

The following metrics are used throughout the project:

Metric	Purpose
Accuracy	Overall classification correctness
Macro F1	Performance across all classes equally
Weighted F1	Class-frequency weighted performance
ROC-AUC	Multiclass discrimination capability
Runtime	Computational efficiency

Macro F1 is particularly important due to class imbalance in several TCGA subtype datasets.

Technologies Used
Programming Languages
Python 3
Core Libraries
pandas
numpy
scikit-learn
matplotlib
Machine Learning Models
Logistic Regression
SGDClassifier
Runtime Considerations

The TCGA datasets contain approximately:

~60,660 gene features

making computational efficiency an important challenge.

Key observations:

Low variance thresholds retain many genes and significantly increase runtime
Moderate feature reduction often improves both runtime and predictive performance
Embedded and wrapper methods are substantially more computationally expensive than filter methods
Example Workflow
Run Variance Threshold Experiments
python variance_filter.py
Run All Filter Methods
python run_filters.py
Generate Embedded Comparison Plots
python generate_comparison_plots.py
Results

The project compares:

predictive accuracy
macro F1 performance
runtime efficiency
dimensionality reduction strength

across multiple feature selection strategies and cancer datasets.

Generated outputs include:

CSV result tables
comparison plots
feature reduction visualizations
ROC-AUC summaries

All generated outputs are saved to the corresponding results/ directories.

Future Work

Potential future improvements include:

Wrapper feature selection methods
Hyperparameter optimization
Cross-validation experiments
Deep learning models
PCA and nonlinear dimensionality reduction
Precision-Recall analysis
Biological pathway interpretation of selected genes
Authors

