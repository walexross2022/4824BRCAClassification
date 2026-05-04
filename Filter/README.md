Runtime Expectations
Approximate runtime on the BRCA dataset (60,660 genes, 1,095 samples):


app.py → ~5–10 seconds


run_filters.py → ~5–7 minutes total (runs all methods and produces summary tables, comparisons, and plots)


Overview
The Filter/ section implements lightweight statistical feature-selection methods for reducing gene dimensionality before classification. These methods provide fast, interpretable benchmarks that can be directly compared against the Baseline/ and Embedded/ approaches in this project.
Unlike embedded methods (such as LASSO), filter methods evaluate each feature independently of the classifier. This makes them faster, easier to interpret, and easier to compare across methods.
Implemented Methods
Variance Threshold
Removes genes with low variance across samples. Genes with little variation are unlikely to contribute useful signal and can be removed as a simple first-pass dimensionality reduction step.
This method was implemented as a threshold sweep using:


0.01


0.05


0.10


0.25


0.50


0.75


This sweep showed that moderate variance pruning improved both runtime and predictive performance, with the strongest overall performance observed near 0.25.
ANOVA / F-test
Ranks genes by how strongly their expression differs across subtype classes using ANOVA F-statistics.
This method retained the top 5,000 genes and provided the strongest stable fixed top-k statistical ranking baseline.
Mutual Information
Ranks genes based on how much information they provide about subtype labels.
This method also retained the top 5,000 genes and serves as the nonlinear ranking benchmark. In the current BRCA setting, Mutual Information performed competitively, but results were less stable across repeated runs than ANOVA. For this reason, Mutual Information is best interpreted as a useful nonlinear comparison method rather than the most reliable top-performing filter.
Methodology
All filter methods use:


one stratified 70/30 train-test split


random_state=42


feature selection on the training set only


scaling after feature selection


SGDClassifier(loss="log_loss") for classification


This keeps all filter methods directly comparable under the same train/test conditions.
Evaluation
Each filter method is evaluated using:


Accuracy


Macro F1


Weighted F1


Runtime


Current Findings
Current BRCA results suggest:


Variance Threshold (tuned) provides the strongest overall performance


ANOVA / F-test provides the strongest stable fixed top-k ranking method


Mutual Information serves as a useful nonlinear benchmark, but is less stable across repeated runs


The strongest observed variance threshold was approximately 0.25, which produced the best balance of:


feature reduction


runtime


predictive performance


Files


app.py → shared BRCA data loader


variance_filter.py → variance threshold sweep


anova_filter.py → ANOVA feature ranking


mutual_info_filter.py → Mutual Information feature ranking


run_filters.py → runs all filter methods and aggregates results


results/ → stores metrics, plots, and summaries

