# TCGA Cancer Subtype Classification Using Machine Learning

This project investigates the use of Machine Learning methods for multiclass cancer subtype classification using high-dimensional gene expression data from The Cancer Genome Atlas (TCGA). The primary goal is to compare baseline, filter-based, wrapper-based, and embedded feature selection approaches for reducing dimensionality while maintaining strong subtype classification performance.

The project evaluates multiple machine learning pipelines across three TCGA cancer datasets:

- BRCA — Breast Invasive Carcinoma
- COAD — Colon Adenocarcinoma
- PRAD — Prostate Adenocarcinoma

Each dataset contains approximately 60,660 gene expression features per sample, making dimensionality reduction and feature selection essential components of the workflow.

---

# Project Goals

The primary objectives of this project are:

- Perform multiclass cancer subtype classification using gene expression data
- Compare Filter, Wrapper, and Embedded feature selection strategies
- Evaluate dimensionality reduction effectiveness
- Analyze tradeoffs between runtime, interpretability, and predictive performance
- Investigate how aggressive feature reduction affects classification quality in imbalanced datasets
- Compare reduced-feature models against a full-feature baseline classifier

---

# Datasets

The project uses TCGA gene expression matrices and associated subtype labels.

## Included Cancer Types

| Dataset | Samples | Subtypes | Notes |
|---|---|---|---|
| BRCA | ~1,231 | 5 | Breast cancer molecular subtypes |
| COAD | ~524 | 4 | Colon cancer subtype classification |
| PRAD | ~554 | 8 | High class imbalance and minority subtype difficulty |

Each dataset includes:

- Gene expression matrix
- Patient subtype labels
- Sample alignment preprocessing
- Duplicate patient handling

---

# Project Structure

- `Baseline/`
  - Baseline Logistic Regression experiments using the complete feature set
  - Includes multicancer comparison plots and baseline metrics

- `Filter/`
  - Statistical feature selection methods
  - Includes:
    - Variance Threshold
    - ANOVA / F-test
    - Mutual Information
  - Contains preprocessing loader (`app.py`) and automated benchmark scripts

- `Embedded/`
  - Model-integrated feature selection methods
  - Includes:
    - LASSO Logistic Regression
    - Random Forest Feature Importance

- `Wrapper/`
  - Search-based feature selection approaches
  - Includes:
    - SelectKBest
    - Forward Selection with variance pre-filtering

- `data/`
  - Local directory containing TCGA expression matrices and subtype labels

- `results/`
  - Generated CSV summaries
  - Benchmark plots
  - Comparison visualizations
  - ROC-AUC analyses

---

# Environment Setup

This project was developed using:

- Python 3.9+
- Windows 11
- scikit-learn
- pandas
- numpy
- matplotlib

Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

# Data Placement

Place all TCGA `.csv` or `.parquet` files inside the `data/` directory using the following structure:

```text
data/
├── TCGA-BRCA/
├── TCGA-COAD/
└── TCGA-PRAD/
```

Each dataset directory should contain:

- Expression matrix
- Subtype label file

---

# How to Reproduce Results

To reproduce the experiments and figures used in the final report, run the following scripts.

## 1. Run Baseline Experiments

```bash
python Baseline/baseline_run.py
```

This generates:

- Baseline Logistic Regression metrics
- Cross-cancer comparison plots
- ROC-AUC summaries

---

## 2. Run Filter Method Benchmarks

```bash
python Filter/run_filters.py
```

This evaluates:

- Variance Threshold sweeps
- ANOVA / F-test selection
- Mutual Information selection

Outputs include:

- Accuracy
- Macro F1
- ROC-AUC
- Runtime
- Feature count comparisons

---

## 3. Run Wrapper Methods

```bash
python Wrapper/select_k_best.py
```

Wrapper methods include:

- SelectKBest
- Forward Selection

Forward Selection experiments use aggressive pre-filtering due to the computational cost of operating on 60,660-dimensional datasets.

---

## 4. Run Embedded Methods

```bash
python Embedded/lasso_experiments.py
```

Embedded experiments include:

- LASSO Logistic Regression
- Random Forest Feature Importance

---

## 5. Generate Comparison Visualizations

```bash
python results/generate_comparison_plots.py
```

This generates:

- Accuracy comparison plots
- Macro F1 comparison plots
- ROC-AUC comparison plots
- Feature reduction visualizations

---

# Methodology

## 1. Baseline Models

Baseline experiments were performed using the full feature set (`p = 60,660`) with Logistic Regression using the SAGA solver. These models establish a reference performance ceiling for comparison against reduced-feature methods.

---

## 2. Filter Methods

Filter methods evaluate features independently of the classifier.

### Implemented Methods

- Variance Threshold
  - Threshold sweep from `0.05` to `0.75`

- ANOVA / F-test
  - Selects statistically significant genes

- Mutual Information
  - Captures nonlinear relationships between genes and cancer subtypes

Filter methods provide the best balance between computational efficiency and predictive performance.

---

## 3. Wrapper Methods

Wrapper methods iteratively search for feature subsets using model feedback.

### Implemented Methods

- SelectKBest
- Forward Selection

Due to the extremely high dimensionality of TCGA datasets, wrapper methods require heavy pre-filtering and remain computationally expensive.

---

## 4. Embedded Methods

Embedded methods perform feature selection during model training.

### Implemented Methods

- LASSO Logistic Regression (L1 Regularization)
- Random Forest Feature Importance

These methods attempt to automatically identify informative genes while optimizing classifier performance.

---

# Evaluation Metrics

The following metrics are used throughout the project:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Macro F1 | Balanced performance across all classes |
| Weighted F1 | Frequency-weighted classification performance |
| ROC-AUC | Multiclass discrimination capability |
| Runtime | Computational efficiency |

Macro F1 is particularly important for PRAD due to severe subtype imbalance.

---

# Key Findings

## Baseline Paradox

The full-feature baseline frequently establishes a performance ceiling that aggressive feature selection methods struggle to surpass.

---

## Dimensionality Reduction Efficiency

Filter methods such as ANOVA and Mutual Information achieve:

- 90%+ feature reduction
- competitive ROC-AUC
- strong classification accuracy

while remaining computationally manageable.

---

## Runtime Tradeoffs

- Variance Threshold is extremely fast but often retains many features
- Mutual Information produces strong results but incurs heavy runtime costs
- Wrapper methods become computationally infeasible without aggressive pre-filtering

---

## Class Imbalance Challenges

PRAD consistently remains the most difficult dataset.

Although many methods achieve relatively high Accuracy, Macro F1 scores remain low because classifiers frequently fail to identify rare cancer subtypes.

This highlights the importance of evaluating:

- Macro F1
- ROC-AUC
- minority subtype performance

rather than relying solely on overall Accuracy.

---

# Example Results

## Baseline Multi-Cancer Comparison

The project includes aggregated comparison plots across BRCA, COAD, and PRAD datasets using repeated baseline Logistic Regression experiments.

### Aggregated Metric Comparison

![Aggregated Comparison](Baseline/results/Comparison/cross_cancer_aggregated_comparison.png)

### Per-Metric Cancer Comparison

![Per-Metric Comparison](Baseline/results/Comparison/cross_cancer_per_metric_comparison.png)

---

# Future Work

Potential future improvements include:

- Recursive Feature Elimination (RFE)
- Hyperparameter optimization
- Cross-validation studies
- Precision-Recall analysis
- PCA and nonlinear dimensionality reduction
- Deep learning architectures
- Biological pathway interpretation of selected genes

---

# Authors

- William Alex Ross
- Qianhe (Ted) Sha
- Bryan Pham
- Yassin Lahrime

Virginia Tech Machine Learning Project  
TCGA Cancer Subtype Classification using Machine Learning and Feature Selection Techniques
