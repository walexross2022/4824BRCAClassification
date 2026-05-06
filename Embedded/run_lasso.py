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

# Configuration
BASE_OUT_DIR = "Embedded/results/LASSO"
RANDOM_STATE = 42

def run_lasso_workflow(cancer_type):
    print(f"\n=== Starting LASSO Workflow for {cancer_type} ===")
    
    # 1. Load Data
    X_raw, y, labels = load_tcga_data(cancer_type)
    feature_names = X_raw.columns
    
    # Create timestamped results directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(BASE_OUT_DIR, f"{cancer_type}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    # ... (Step 2-4 remains same) ...
    # 5. Feature Selection & Model Training (with K-fold Cross-Validation)
    print(f"[{cancer_type}] Training LASSO with 3-fold CV...")
    start_time = time.time()
    
    # LogisticRegressionCV automates the search for the best C using stratified K-fold
    # Optimizing for speed: 3-fold CV, 5 Cs, higher tolerance, and verbose output
    model_params = {
        'penalty': 'l1',
        'solver': 'saga',
        'cv': 3,
        'Cs': 5,
        'tol': 0.1,
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': 1
    }
    
    model = LogisticRegressionCV(
        penalty=model_params['penalty'],
        solver=model_params['solver'],
        cv=StratifiedKFold(model_params['cv'], shuffle=True, random_state=RANDOM_STATE),
        Cs=model_params['Cs'],
        tol=model_params['tol'],
        scoring='f1_macro',
        max_iter=model_params['max_iter'],
        random_state=model_params['random_state'],
        n_jobs=model_params['n_jobs'],
        verbose=model_params['verbose']
    )
    model.fit(X_train, y_train)
    
    fs_runtime = time.time() - start_time
    print(f"[{cancer_type}] Best C identified: {model.C_[0]:.4f}")
    
    # 6. Feature Selection Analysis
    # In multiclass (OvR), we consider a feature selected if it's non-zero for ANY class
    selected_mask = np.any(model.coef_ != 0, axis=0)
    n_selected = np.sum(selected_mask)
    selected_features = selected_features_pre[selected_mask].tolist()
    
    # 7. Final Evaluation on Test Set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    # Multi-class ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=labels)
    # If binary classification, y_prob might need slicing, but for multi-class it's (n_samples, n_classes)
    if len(labels) == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
        
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"[{cancer_type}] Results: Acc={acc:.4f}, F1={f1_macro:.4f}, ROC-AUC={roc_auc:.4f}")
    
    # Save Results
    with open(os.path.join(out_dir, f"{cancer_type}_lasso_report.txt"), "w") as f:
        f.write(f"=== {cancer_type} LASSO Report ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"--- Model Parameters ---\n")
        for k, v in model_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Best C identified: {model.C_[0]:.4f}\n\n")
        
        f.write(f"--- Metrics ---\n")
        f.write(f"Selected Features: {n_selected}\n")
        f.write(f"FS/Training Runtime: {fs_runtime:.2f}s\n")
        f.write(f"Accuracy:          {acc:.4f}\n")
        f.write(f"Macro F1:          {f1_macro:.4f}\n")
        f.write(f"Weighted ROC-AUC:  {roc_auc:.4f}\n")
        f.write(f"Macro Precision:   {precision:.4f}\n")
        f.write(f"Macro Recall:      {recall:.4f}\n\n")
        f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
        f.write(f"\nTop Selected Features:\n")
        f.write(", ".join(selected_features[:50]))
        if len(selected_features) > 50:
            f.write(f" ... (+{len(selected_features)-50} more)")
            
    # Visualizations
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title(f"{cancer_type} - LASSO Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    
    return {
        "cancer": cancer_type,
        "n_features": n_selected,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc
    }

if __name__ == "__main__":
    summary = []
    for cancer in ["BRCA", "COAD", "PRAD"]:
        res = run_lasso_workflow(cancer)
        summary.append(res)
    
    pd.DataFrame(summary).to_csv(os.path.join(BASE_OUT_DIR, "summary_metrics.csv"), index=False)
    print("\n[DONE] All LASSO runs completed.")
