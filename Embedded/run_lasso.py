import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Allow importing Filter/app.py for cached dataset loading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Filter"))
from app import load_dataset_cached

# Configuration
BASE_OUT_DIR = "Embedded/results/LASSO"
RANDOM_STATE = 42
N_RUNS = 10

def run_lasso_workflow(cancer_type, seed, save_plots=True, use_cache=True, n_jobs=-1):
    print(f"\n=== Starting LASSO Workflow for {cancer_type} (Seed {seed}) ===")
    
    if use_cache:
        X_raw_df, y = load_dataset_cached(cancer_type)
        labels = sorted(y.unique())
        feature_names = X_raw_df.columns
    else:
        X_raw_df, y, labels = load_tcga_data(cancer_type)
        feature_names = X_raw_df.columns
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw_df, y, test_size=0.3, stratify=y, random_state=seed
    )
    
    selector = VarianceThreshold(threshold=0.0) 
    X_train_pre = selector.fit_transform(X_train_raw)
    selected_features_pre = feature_names[selector.get_support()]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pre)
    
    print(f"[{cancer_type}] Training LASSO to identify features...")
    lasso = LogisticRegressionCV(
        penalty='l1', solver='saga', cv=3, Cs=5, tol=0.1,
        max_iter=1000, random_state=seed, n_jobs=n_jobs
    )
    lasso.fit(X_train_scaled, y_train)
    
    selected_mask = np.any(lasso.coef_ != 0, axis=0)
    selected_genes = selected_features_pre[selected_mask]
    
    X_selected = X_raw_df[selected_genes]
    
    results = evaluate_feature_set(
        X_selected, y, "LASSO", cancer_type, labels, 
        os.path.join(BASE_OUT_DIR, cancer_type),
        seed=seed,
    )
    results["seed"] = seed
    return results

if __name__ == "__main__":
    cancers = ["BRCA", "COAD", "PRAD"]
    rng = np.random.default_rng(RANDOM_STATE)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=N_RUNS)]
    metrics_cols = ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "weighted_recall", "roc_auc"]
    
    for cancer in cancers:
        os.makedirs(os.path.join(BASE_OUT_DIR, cancer), exist_ok=True)
        all_runs = []
        
        for run_idx, seed in enumerate(seeds):
            res = run_lasso_workflow(cancer, seed)
            res["run"] = run_idx + 1
            all_runs.append(res)
            print(f"  [{cancer}] Run {run_idx+1}/{N_RUNS} (seed={seed}) - Acc: {res['accuracy']:.4f}, Macro F1: {res['macro_f1']:.4f}, ROC-AUC: {res['roc_auc']:.4f}")
        
        df_runs = pd.DataFrame(all_runs)
        df_runs.to_csv(os.path.join(BASE_OUT_DIR, cancer, f"{cancer}_run_results.csv"), index=False)
        
        summary = {col: {"mean": df_runs[col].mean(), "std": df_runs[col].std()} for col in metrics_cols}
        summary["n_features"] = {"mean": df_runs["n_features"].mean(), "std": df_runs["n_features"].std()}
        summary["train_time"] = {"mean": df_runs["train_time"].mean(), "std": df_runs["train_time"].std()}
        
        pd.DataFrame(summary).to_csv(os.path.join(BASE_OUT_DIR, cancer, f"{cancer}_summary.csv"))
        
        print(f"\n  [{cancer}] 10-run summary (mean +/- std):")
        for col in metrics_cols:
            print(f"    {col}: {summary[col]['mean']:.4f} +/- {summary[col]['std']:.4f}")
        print(f"    n_features: {summary['n_features']['mean']:.1f} +/- {summary['n_features']['std']:.1f}")
    
    print("\n[DONE] All LASSO runs completed (10 runs per cancer).")
