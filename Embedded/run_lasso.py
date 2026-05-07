import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_auc_score, precision_recall_fscore_support
)
from sklearn.feature_selection import VarianceThreshold

from Embedded.data_utils import load_tcga_data
from shared_utils import evaluate_feature_set

# Configuration
BASE_OUT_DIR = "Embedded/results/LASSO"
RANDOM_STATE = 42

def run_lasso_workflow(cancer_type):
    print(f"\n=== Starting LASSO Workflow for {cancer_type} ===")
    
    # 1. Load Data
    X_raw, y, labels = load_tcga_data(cancer_type)
    feature_names = X_raw.columns
    
    # 2. Train / Test Split for FS
    # We split here to ensure FS only happens on training data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    
    # 3. Pre-filtering & Scaling
    selector = VarianceThreshold(threshold=0.0) 
    X_train_pre = selector.fit_transform(X_train_raw)
    selected_features_pre = feature_names[selector.get_support()]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pre)
    
    # 4. Feature Selection (LASSO)
    print(f"[{cancer_type}] Training LASSO to identify features...")
    lasso = LogisticRegressionCV(
        penalty='l1', solver='saga', cv=3, Cs=5, tol=0.1,
        max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
    )
    lasso.fit(X_train_scaled, y_train)
    
    # Identify selected features
    selected_mask = np.any(lasso.coef_ != 0, axis=0)
    selected_genes = selected_features_pre[selected_mask]
    
    # 5. Standardized Evaluation
    # We pass the subset of RAW features to the shared evaluator
    # to maintain pipeline consistency (split/scale/train)
    X_selected = X_raw[selected_genes]
    
    results = evaluate_feature_set(
        X_selected, y, "LASSO", cancer_type, labels, 
        os.path.join(BASE_OUT_DIR, cancer_type)
    )
    
    return results

if __name__ == "__main__":
    summary = []
    for cancer in ["BRCA", "COAD", "PRAD"]:
        res = run_lasso_workflow(cancer)
        summary.append(res)
    
    pd.DataFrame(summary).to_csv(os.path.join(BASE_OUT_DIR, "summary_metrics.csv"), index=False)
    print("\n[DONE] All LASSO runs completed.")
