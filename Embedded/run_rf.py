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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_auc_score, precision_recall_fscore_support
)
from sklearn.feature_selection import VarianceThreshold

from Embedded.data_utils import load_tcga_data
from shared_utils import evaluate_feature_set

# Configuration
BASE_OUT_DIR = "Embedded/results/RandomForest"
RANDOM_STATE = 42

def run_rf_workflow(cancer_type):
    print(f"\n=== Starting Random Forest Workflow for {cancer_type} ===")
    
    # 1. Load Data
    X_raw, y, labels = load_tcga_data(cancer_type)
    feature_names = X_raw.columns
    
    # 2. Train / Test Split for FS
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    
    # 3. Pre-filtering & Scaling
    selector = VarianceThreshold(threshold=0.0) 
    X_train_pre = selector.fit_transform(X_train_raw)
    selected_features_pre = feature_names[selector.get_support()]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pre)
    
    # 4. Feature Selection (Random Forest)
    print(f"[{cancer_type}] Tuning Random Forest to identify features...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf = grid_search.best_estimator_
    
    # Identify selected features (importance > mean)
    importances = best_rf.feature_importances_
    selected_mask = importances > np.mean(importances)
    selected_genes = selected_features_pre[selected_mask]
    
    # 5. Standardized Evaluation
    X_selected = X_raw[selected_genes]
    
    results = evaluate_feature_set(
        X_selected, y, "RandomForest", cancer_type, labels, 
        os.path.join(BASE_OUT_DIR, cancer_type)
    )
    
    return results

if __name__ == "__main__":
    summary = []
    for cancer in ["BRCA", "COAD", "PRAD"]:
        res = run_rf_workflow(cancer)
        summary.append(res)
    
    pd.DataFrame(summary).to_csv(os.path.join(BASE_OUT_DIR, "summary_metrics.csv"), index=False)
    print("\n[DONE] All Random Forest runs completed.")
