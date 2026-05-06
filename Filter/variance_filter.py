# Filter/variance_filter.py

import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Filter", "results")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]


# -----------------------------
# Main Runner
# -----------------------------
def run_variance_filter(cancer_type="BRCA"):
    cancer_type = cancer_type.upper()

    # -----------------------------
    # STEP 0: Load Dataset
    # -----------------------------
    X, y = load_dataset(cancer_type)

    # -----------------------------
    # STEP 1: Train / Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"\n[{cancer_type}] Train shape: {X_train.shape}")
    print(f"[{cancer_type}] Test shape:  {X_test.shape}")

    # -----------------------------
    # STEP 2: Run Threshold Sweep
    # -----------------------------
    all_results = []

    for threshold in THRESHOLDS:
        print("\n" + "=" * 50)
        print(f"[{cancer_type}] Running Variance Threshold = {threshold}")
        print("=" * 50)

        start_time = time.time()

        selector = VarianceThreshold(threshold=threshold)
        X_train_selected = selector.fit_transform(X_train)
        X_test_selected = selector.transform(X_test)

        selected_features = X_train_selected.shape[1]

        print(f"Original features: {X.shape[1]}")
        print(f"Selected features: {selected_features}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        print("Starting classifier training...")
        train_start = time.time()

        model = SGDClassifier(loss="log_loss", random_state=RANDOM_STATE)
        classes = y_train.unique()

        for epoch in range(50):
            model.partial_fit(X_train_scaled, y_train, classes=classes)

        train_time = time.time() - train_start
        print(f"Training finished in {train_time:.2f} seconds")

        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        runtime = time.time() - start_time

        print(f"Accuracy:    {acc:.4f}")
        print(f"Macro F1:    {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Runtime (s): {runtime:.2f}")

        all_results.append({
            "Cancer": cancer_type,
            "Method": "Variance Threshold",
            "Threshold": threshold,
            "Original Features": X.shape[1],
            "Selected Features": selected_features,
            "Accuracy": acc,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "Runtime (s)": runtime
        })

    results = pd.DataFrame(all_results)

    # Save per-dataset CSV
    results.to_csv(
        os.path.join(OUT_DIR, f"{cancer_type.lower()}_variance_filter_results.csv"),
        index=False
    )

    # Save threshold comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["Threshold"], results["Accuracy"], marker="o", linewidth=2, label="Accuracy")
    plt.plot(results["Threshold"], results["Macro F1"], marker="o", linewidth=2, label="Macro F1")
    plt.plot(results["Threshold"], results["Weighted F1"], marker="o", linewidth=2, label="Weighted F1")

    plt.title(f"{cancer_type} - Variance Threshold Sweep")
    plt.xlabel("Variance Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cancer_type.lower()}_variance_threshold_sweep.png"), dpi=150)
    plt.close()

    # Save feature count plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["Threshold"], results["Selected Features"], marker="o", linewidth=2)

    plt.title(f"{cancer_type} - Selected Features vs Variance Threshold")
    plt.xlabel("Variance Threshold")
    plt.ylabel("Selected Features")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cancer_type.lower()}_variance_selected_features.png"), dpi=150)
    plt.close()

    print(f"\n[{cancer_type}] Done - variance threshold sweep saved to {OUT_DIR}/")
    return results


# -----------------------------
# Run All + Save Combined TXT
# -----------------------------
if __name__ == "__main__":
    brca_results = run_variance_filter("BRCA")
    coad_results = run_variance_filter("COAD")
    prad_results = run_variance_filter("PRAD")

    all_results = pd.concat([brca_results, coad_results, prad_results], ignore_index=True)

    with open(os.path.join(OUT_DIR, "variance_results.txt"), "w") as f:
        f.write("=== Variance Threshold Sweep (All Datasets) ===\n\n")
        f.write(all_results.to_string(index=False))

        for cancer in ["BRCA", "COAD", "PRAD"]:
            subset = all_results[all_results["Cancer"] == cancer]

            best_acc = subset.loc[subset["Accuracy"].idxmax()]
            best_f1 = subset.loc[subset["Macro F1"].idxmax()]
            best_rt = subset.loc[subset["Runtime (s)"].idxmin()]

            f.write(f"\n\n=== {cancer} Summary ===\n")

            f.write(
                f"\nBest Accuracy Threshold: {best_acc['Threshold']}"
                f"\n  Accuracy: {best_acc['Accuracy']:.4f}"
                f"\n  Macro F1: {best_acc['Macro F1']:.4f}"
                f"\n  Runtime (s): {best_acc['Runtime (s)']:.2f}\n"
            )

            f.write(
                f"\nBest Macro F1 Threshold: {best_f1['Threshold']}"
                f"\n  Accuracy: {best_f1['Accuracy']:.4f}"
                f"\n  Macro F1: {best_f1['Macro F1']:.4f}"
                f"\n  Runtime (s): {best_f1['Runtime (s)']:.2f}\n"
            )

            f.write(
                f"\nFastest Runtime Threshold: {best_rt['Threshold']}"
                f"\n  Accuracy: {best_rt['Accuracy']:.4f}"
                f"\n  Macro F1: {best_rt['Macro F1']:.4f}"
                f"\n  Runtime (s): {best_rt['Runtime (s)']:.2f}\n"
            )