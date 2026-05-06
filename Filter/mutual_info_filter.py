# Filter/mutual_info_filter.py

import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Filter", "results")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TOP_K = 5000


# -----------------------------
# Main Runner
# -----------------------------
def run_mutual_info_filter(cancer_type="BRCA"):
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
    # STEP 2: Mutual Information Feature Selection
    # -----------------------------
    start_time = time.time()

    selector = SelectKBest(score_func=mutual_info_classif, k=TOP_K)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train_selected.shape[1]

    print(f"\n[{cancer_type}] Mutual Information Feature Selection")
    print(f"Top K Features:     {TOP_K}")
    print(f"Original features:  {X.shape[1]}")
    print(f"Selected features:  {selected_features}")

    # -----------------------------
    # STEP 3: Scale Reduced Features
    # -----------------------------
    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # -----------------------------
    # STEP 4: Train Classifier
    # -----------------------------
    print("\nStarting classifier training...")
    train_start = time.time()

    model = SGDClassifier(loss="log_loss", random_state=RANDOM_STATE)
    classes = y_train.unique()

    for epoch in range(50):
        model.partial_fit(X_train_selected, y_train, classes=classes)

    print(f"Training finished in {time.time() - train_start:.2f} seconds")

    # -----------------------------
    # STEP 5: Evaluate
    # -----------------------------
    y_pred = model.predict(X_test_selected)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    runtime = time.time() - start_time

    print(f"\nAccuracy:      {acc:.4f}")
    print(f"Macro F1:      {f1_macro:.4f}")
    print(f"Weighted F1:   {f1_weighted:.4f}")
    print(f"Runtime (s):   {runtime:.2f}")

    # -----------------------------
    # STEP 6: Confusion Matrix
    # -----------------------------
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{cancer_type} - Mutual Information Filter Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cancer_type.lower()}_mutual_info_confusion_matrix.png"), dpi=150)
    plt.close()

    # -----------------------------
    # STEP 7: Metrics Bar Chart
    # -----------------------------
    plt.figure(figsize=(8, 5))
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    values = [acc, f1_macro, f1_weighted]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    bars = plt.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.ylim(0, 1.1)
    plt.title(f"{cancer_type} - Mutual Information Filter Performance")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cancer_type.lower()}_mutual_info_metrics_bar.png"), dpi=150)
    plt.close()

    # -----------------------------
    # STEP 8: Save Results
    # -----------------------------
    results = pd.DataFrame([{
        "Cancer": cancer_type,
        "Method": "Mutual Information",
        "Top K": TOP_K,
        "Original Features": X.shape[1],
        "Selected Features": selected_features,
        "Accuracy": acc,
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "Runtime (s)": runtime
    }])

    results.to_csv(
        os.path.join(OUT_DIR, f"{cancer_type.lower()}_mutual_info_filter_results.csv"),
        index=False
    )

    print(f"\n[{cancer_type}] Done - results saved to {OUT_DIR}/")
    return results, classification_report(y_test, y_pred)


# -----------------------------
# Run All + Save Combined TXT
# -----------------------------
if __name__ == "__main__":
    brca_results, brca_report = run_mutual_info_filter("BRCA")
    coad_results, coad_report = run_mutual_info_filter("COAD")
    prad_results, prad_report = run_mutual_info_filter("PRAD")

    all_results = pd.concat([brca_results, coad_results, prad_results], ignore_index=True)

    with open(os.path.join(OUT_DIR, "mutual_info_results.txt"), "w") as f:
        f.write("=== Mutual Information Filter (All Datasets) ===\n\n")
        f.write(all_results.to_string(index=False))

        reports = {
            "BRCA": brca_report,
            "COAD": coad_report,
            "PRAD": prad_report
        }

        for cancer in ["BRCA", "COAD", "PRAD"]:
            subset = all_results[all_results["Cancer"] == cancer].iloc[0]

            f.write(f"\n\n=== {cancer} Summary ===\n")
            f.write(f"\nTop K:         {int(subset['Top K'])}")
            f.write(f"\nAccuracy:      {subset['Accuracy']:.4f}")
            f.write(f"\nMacro F1:      {subset['Macro F1']:.4f}")
            f.write(f"\nWeighted F1:   {subset['Weighted F1']:.4f}")
            f.write(f"\nRuntime (s):   {subset['Runtime (s)']:.2f}\n")

            f.write(f"\nClassification Report:\n{reports[cancer]}\n")