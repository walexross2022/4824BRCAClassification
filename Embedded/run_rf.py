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

# Configuration
BASE_OUT_DIR = "Embedded/results/RandomForest"
RANDOM_STATE = 42

def run_rf_workflow(cancer_type):
    print(f"\n=== Starting Random Forest Workflow for {cancer_type} ===")
    
    # 1. Load Data
    X_raw, y, labels = load_tcga_data(cancer_type)
    feature_names = X_raw.columns
    out_dir = os.path.join(BASE_OUT_DIR, cancer_type)
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Basic Cleaning (Gene Filtering)
    var_selector = VarianceThreshold(threshold=0.01)
    X_filtered = var_selector.fit_transform(X_raw)
    selected_features_pre = feature_names[var_selector.get_support()]
    print(f"[{cancer_type}] Genes after variance filter: {X_filtered.shape[1]} / {X_raw.shape[1]}")

    # 3. Data Partitioning (80/20 split)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_filtered, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    
    # 4. Normalization (On Training Set Only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 5. Feature Selection & Model Training (Random Forest Importance)
    print(f"[{cancer_type}] Tuning Random Forest via Grid Search...")
    start_time = time.time()
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    runtime = time.time() - start_time
    print(f"[{cancer_type}] Best Params: {grid_search.best_params_}")
    
    # 6. Feature Importance Selection
    # Select features that contribute more than the mean importance
    importances = best_rf.feature_importances_
    mean_importance = np.mean(importances)
    selected_mask = importances > mean_importance
    n_selected = np.sum(selected_mask)
    selected_features = selected_features_pre[selected_mask].tolist()
    
    # 7. Evaluation
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    y_test_bin = label_binarize(y_test, classes=labels)
    if len(labels) == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
    
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"[{cancer_type}] Results: Acc={acc:.4f}, F1={f1_macro:.4f}, ROC-AUC={roc_auc:.4f}")
    
    # Save Results
    with open(os.path.join(out_dir, f"{cancer_type}_rf_report.txt"), "w") as f:
        f.write(f"=== {cancer_type} Random Forest (Professor's Workflow) ===\n\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")
        f.write(f"Features with Importance > Mean: {n_selected}\n")
        f.write(f"Runtime: {runtime:.2f}s\n\n")
        f.write(f"Accuracy:          {acc:.4f}\n")
        f.write(f"Macro F1:          {f1_macro:.4f}\n")
        f.write(f"Weighted ROC-AUC:  {roc_auc:.4f}\n")
        f.write(f"Macro Precision:   {precision:.4f}\n")
        f.write(f"Macro Recall:      {recall:.4f}\n\n")
        f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
        f.write(f"\nTop 50 Most Important Features:\n")
        sorted_indices = np.argsort(importances)[::-1]
        top_features = selected_features_pre[sorted_indices[:50]]
        f.write(", ".join(top_features))
            
    # Visualizations
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", xticklabels=labels, yticklabels=labels)
    plt.title(f"{cancer_type} - Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    
    return {
        "cancer": cancer_type,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc
    }

if __name__ == "__main__":
    summary = []
    for cancer in ["BRCA", "COAD", "PRAD"]:
        res = run_rf_workflow(cancer)
        summary.append(res)
    
    pd.DataFrame(summary).to_csv(os.path.join(BASE_OUT_DIR, "summary_metrics.csv"), index=False)
    print("\n[DONE] All Random Forest runs completed.")
