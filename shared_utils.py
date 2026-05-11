import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_auc_score, recall_score
)
from sklearn.preprocessing import label_binarize

RANDOM_STATE = 42

def evaluate_feature_set(X_selected, y, method_name, cancer_type, labels, output_dir, seed=None):
    """
    Standardized evaluation pipeline for all feature selection methods.
    Ensures that the final classification is always done using the same model
    and parameters, regardless of how features were selected.
    """
    print(f"[{method_name} - {cancer_type}] Starting standardized evaluation...")

    eval_seed = RANDOM_STATE if seed is None else int(seed)

    # 1. Standard Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, 
        test_size=0.3, 
        stratify=y, 
        random_state=eval_seed
    )

    # 2. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Standard Classifier (Project-wide consistency)
    # Logistic Regression with L-BFGS solver (matches baseline)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=eval_seed,
        n_jobs=-1,
    )
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    # 4. Standard Metrics
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Standardized ROC-AUC calculation (with NaN safety)
    try:
        y_test_bin = label_binarize(y_test, classes=labels)
        y_prob_finite = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
        if len(labels) == 2:
            roc_auc = roc_auc_score(y_test, y_prob_finite[:, 1])
        else:
            roc_auc = roc_auc_score(y_test_bin, y_prob_finite, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"  WARNING: ROC-AUC calculation failed for {method_name}/{cancer_type}: {e}")
        roc_auc = float("nan")

    # 5. Save Standardized Report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{method_name.lower()}_results.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Standardized Report: {method_name} on {cancer_type}\n")
        f.write(f"Features: {X_selected.shape[1]}\n")
        f.write(f"Training Time: {train_time:.2f}s\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n")
        f.write(f"Macro Recall: {recall_macro:.4f}\n")
        f.write(f"Weighted Recall: {recall_weighted:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

    return {
        "Method": method_name,
        "Dataset": cancer_type,
        "n_features": X_selected.shape[1],
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_recall": recall_macro,
        "weighted_recall": recall_weighted,
        "roc_auc": roc_auc,
        "train_time": train_time
    }
