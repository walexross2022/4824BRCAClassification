# Wrapper/selectkbest_wrapper.py

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import softmax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing Filter/app.py for cached dataset loading
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Filter"))
sys.path.insert(0, BASE_DIR)

from app import load_dataset, load_dataset_cached

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -----------------------------
# Configuration
# -----------------------------
OUT_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
FEATURE_FRACTIONS = [0.50]


# -----------------------------
# Main Runner
# -----------------------------
def run_selectkbest_wrapper(
    cancer_type="PRAD",
    seed=RANDOM_STATE,
    fractions=None,
    save_plots=True,
    use_cache=True,
    n_jobs=-1,
):
    cancer_type = cancer_type.upper()
    fractions = fractions if fractions is not None else FEATURE_FRACTIONS

    # -----------------------------
    # STEP 0: Load Dataset
    # -----------------------------
    if use_cache:
        X, y = load_dataset_cached(cancer_type)
    else:
        X, y = load_dataset(cancer_type)

    # -----------------------------
    # STEP 1: Train / Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    print(f"\n[{cancer_type}] Train shape: {X_train.shape}")
    print(f"[{cancer_type}] Test shape:  {X_test.shape}")

    total_features = X_train.shape[1]
    n_classes = len(y.unique())
    is_multiclass = n_classes > 2

    # -----------------------------
    # STEP 2: Run Select K Best Sweep
    # -----------------------------
    all_results = []

    for fraction in fractions:
        n_features_to_select = max(1, int(total_features * fraction))

        print("\n" + "=" * 50)
        print(f"[{cancer_type}] SelectKBest: fraction={fraction}  "
              f"n_features={n_features_to_select}  seed={seed}")
        print("=" * 50)

        start_time = time.time()

        # Impute NaNs then scale - fit only on train to avoid leakage
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # SelectKBest selection on scaled data
        selector = SelectKBest(f_classif, k=n_features_to_select)
        selector.fit(X_train_scaled, y_train)

        X_train_selected = selector.transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)

        selected_features = X_train_selected.shape[1]

        print(f"Original features:  {total_features}")
        print(f"Selected features:  {selected_features}")

        # Train final classifier on the selected features
        print("Starting classifier training...")
        train_start = time.time()

        model = SGDClassifier(loss="log_loss", random_state=seed)
        classes = y_train.unique()

        for _ in range(50):
            model.partial_fit(X_train_selected, y_train, classes=classes)

        train_time = time.time() - train_start
        print(f"Training finished in {train_time:.2f} seconds")

        # Evaluate
        y_pred = model.predict(X_test_selected)

        # SGDClassifier probabilities can contain NaN/Inf; use softmax on
        # decision scores for guaranteed valid probabilities.
        y_scores = model.decision_function(X_test_selected)
        y_prob = softmax(y_scores, axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        if is_multiclass:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                    average="macro")
        else:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])

        runtime = time.time() - start_time

        print(f"Accuracy:      {acc:.4f}")
        print(f"Macro F1:      {f1_macro:.4f}")
        print(f"Weighted F1:   {f1_weighted:.4f}")
        print(f"ROC-AUC:       {roc_auc:.4f}")
        print(f"Runtime (s):   {runtime:.2f}")

        all_results.append({
            "Cancer": cancer_type,
            "Method": "Select K Best",
            "Feature Fraction": fraction,
            "Original Features": total_features,
            "Selected Features": selected_features,
            "Accuracy": acc,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "ROC-AUC": roc_auc,
            "Runtime (s)": runtime,
            "Seed": seed,
        })

    results = pd.DataFrame(all_results)

    if not save_plots:
        return results

    # -----------------------------
    # STEP 3: Save Results Table
    # -----------------------------
    results.to_csv(
        os.path.join(OUT_DIR, f"{cancer_type.lower()}_selectkbest_results.csv"),
        index=False,
    )

    # -----------------------------
    # STEP 4: Save Feature Count Comparison Plot
    # -----------------------------
    if len(results) > 1:
        plt.figure(figsize=(10, 6))

        plt.plot(results["Selected Features"], results["Accuracy"],
                 marker="o", linewidth=2, label="Accuracy")
        plt.plot(results["Selected Features"], results["Macro F1"],
                 marker="o", linewidth=2, label="Macro F1")
        plt.plot(results["Selected Features"], results["Weighted F1"],
                 marker="o", linewidth=2, label="Weighted F1")
        plt.plot(results["Selected Features"], results["ROC-AUC"],
                 marker="o", linewidth=2, label="ROC-AUC")

        plt.title(f"{cancer_type} - Select K Best Sweep")
        plt.xlabel("Number of Selected Features")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{cancer_type.lower()}_select_k_best_sweep.png"),
            dpi=150,
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(results["Feature Fraction"], results["Selected Features"],
                 marker="o", linewidth=2)
        plt.title(f"{cancer_type} - Selected Features vs Feature Fraction")
        plt.xlabel("Feature Fraction")
        plt.ylabel("Selected Features")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{cancer_type.lower()}_select_k_best_features.png"),
            dpi=150,
        )
        plt.close()

    # -----------------------------
    # STEP 5: Save Summary TXT
    # -----------------------------
    summary_path = os.path.join(
        OUT_DIR, f"{cancer_type.lower()}_select_k_best_results.txt"
    )
    with open(summary_path, "w") as f:
        f.write(f"=== {cancer_type} Select K Best Sweep ===\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\nBest Accuracy (n_features): ")
        f.write(str(results.loc[results["Accuracy"].idxmax(), "Selected Features"]))
        f.write("\nBest Macro F1 (n_features): ")
        f.write(str(results.loc[results["Macro F1"].idxmax(), "Selected Features"]))
        f.write("\nBest ROC-AUC (n_features): ")
        f.write(str(results.loc[results["ROC-AUC"].idxmax(), "Selected Features"]))
        f.write("\nFastest Runtime (n_features): ")
        f.write(str(results.loc[results["Runtime (s)"].idxmin(), "Selected Features"]))

    print(f"\n[{cancer_type}] Done - select K best sweep saved to {OUT_DIR}/")
    return results


# -----------------------------
# Standalone run
# -----------------------------
if __name__ == "__main__":
    for cancer in ["BRCA", "COAD", "PRAD"]:
        run_selectkbest_wrapper(cancer_type=cancer, seed=RANDOM_STATE)
