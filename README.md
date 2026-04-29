# Benchmarking feature selection techniques for cancer subtype classification

William Alex Ross working on Filter Methods:
The Filter/ section will focus on implementing lightweight statistical feature-selection methods that rank genes before model training. These methods are designed to reduce dimensionality quickly and provide a simple comparison against the baseline and embedded approaches.

Planned methods include:
Variance Threshold
Removes genes with very low variance across samples. This serves as a simple first-pass filter to eliminate uninformative features.
ANOVA / F-test Feature Ranking
Scores genes based on how strongly their expression differs across subtype classes, then retains the top-ranked features for classification.
Mutual Information
Ranks genes by how much information they provide about subtype labels, allowing for simple nonlinear feature relevance scoring.

Each method will follow the same workflow:
Apply filtering on the training set only
Select the top-ranked genes
Train the classifier on the reduced feature set
Evaluate performance against the baseline using the same metrics (Accuracy, F1, ROC-AUC, and Precision-Recall)
This section is intended to provide fast, interpretable feature-selection benchmarks that can be directly compared with embedded and wrapper approaches.

### Feature Selection Methods

### Dataset Used
