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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold

# Configuration
DATA_DIR = "data"
BASE_OUT_DIR = "Embedded/results/LASSO"
RANDOM_STATE = 42

def load_data(cancer_type):
    prefix = cancer_type.lower()
    cancer_folder = f"TCGA-{cancer_type.upper()}"
    
    expr_path = os.path.join(DATA_DIR, cancer_folder, f"{prefix}_expression_matrix.csv")
    sub_path = os.path.join(DATA_DIR, cancer_folder, f"{prefix}_subtypes.csv")
    
    print(f"[{cancer_type}] Loading data...")
    expr = pd.read_csv(expr_path, index_col=0)
    sub = pd.read_csv(sub_path)
    
    sub.columns = sub.columns.str.strip()
    sub = sub[["pan.samplesID", "Subtype_Selected"]].dropna()
    sub["sample_id"] = sub["pan.samplesID"].str[:12]
    sub = sub.drop_duplicates(subset="sample_id")
    
    expr.index = expr.index.str[:12]
    expr = expr.groupby(expr.index).mean()
    
    common = expr.index.intersection(sub["sample_id"])
    X = expr.loc[common].values
    y = sub.set_index("sample_id").loc[common, "Subtype_Selected"]
    
    return X, y, sorted(y.unique()), expr.columns

def run_lasso(cancer_type, C=0.1):
    X, y, labels, feature_names = load_data(cancer_type)
    out_dir = os.path.join(BASE_OUT_DIR, cancer_type)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Pre-filtering by variance to remove completely static genes
    # Use a small threshold to just remove zero-variance or near-zero variance features
    var_selector = VarianceThreshold(threshold=0.01) 
    X_filtered = var_selector.fit_transform(X)
    selected_features_pre = feature_names[var_selector.get_support()]
    print(f"[{cancer_type}] Features after variance filter (threshold=0.01): {X_filtered.shape[1]} / {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"[{cancer_type}] Training LASSO (C={C})...")
    start_time = time.time()
    
    # Using liblinear with OvR as it's often much faster than SAGA for sparse L1
    model = LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=C, 
        random_state=RANDOM_STATE, 
        max_iter=1000
    )
    model.fit(X_train, y_train)
    
    runtime = time.time() - start_time
    
    # Feature selection analysis
    # For liblinear OvR, coef_ is (n_classes, n_features)
    selected_mask = np.any(model.coef_ != 0, axis=0)
    n_selected = np.sum(selected_mask)
    selected_features = selected_features_pre[selected_mask].tolist()
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print(f"  Selected features: {n_selected}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro F1:          {f1_macro:.4f}")
    
    # Save Results
    with open(os.path.join(out_dir, f"{cancer_type}_lasso_results.txt"), "w") as f:
        f.write(f"=== {cancer_type} LASSO Feature Selection (C={C}) ===\n\n")
        f.write(f"Original Features: {X.shape[1]}\n")
        f.write(f"Variance Filter:   {X_filtered.shape[1]}\n")
        f.write(f"Selected Features: {n_selected}\n")
        f.write(f"Runtime:           {runtime:.2f}s\n\n")
        f.write(f"Accuracy:          {acc:.4f}\n")
        f.write(f"Macro F1:          {f1_macro:.4f}\n")
        f.write(f"Weighted F1:       {f1_weighted:.4f}\n\n")
        f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
        f.write(f"\nSelected Features List (top 100 max):\n")
        f.write(", ".join(selected_features[:100]))
        if len(selected_features) > 100:
            f.write(f"... and {len(selected_features)-100} more")
            
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(f"{cancer_type} - Confusion Matrix (LASSO C={C})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cancer_type}_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Metrics Bar
    plt.figure(figsize=(8, 5))
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    values = [acc, f1_macro, f1_weighted]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = plt.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, 1.1)
    plt.title(f"{cancer_type} - LASSO Performance")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cancer_type}_metrics_bar.png"), dpi=150)
    plt.close()
    
    return {
        "cancer": cancer_type,
        "n_selected": n_selected,
        "accuracy": acc,
        "f1_macro": f1_macro
    }

if __name__ == "__main__":
    results = []
    for cancer in ["BRCA", "COAD", "PRAD"]:
        res = run_lasso(cancer, C=0.1)
        results.append(res)
        
    print("\n=== Final Summary ===")
    summary_df = pd.DataFrame(results)
    print(summary_df)
    summary_df.to_csv(os.path.join(BASE_OUT_DIR, "lasso_summary.csv"), index=False)
