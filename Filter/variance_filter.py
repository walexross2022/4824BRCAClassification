# Filter/variance_filter.py

import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import X, y

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

CANCER = "BRCA"
RANDOM_STATE = 42
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, .5, .75]

# -----------------------------
# STEP 1: Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# -----------------------------
# STEP 2: Run Threshold Sweep
# -----------------------------
all_results = []

for threshold in THRESHOLDS:
    print("\n" + "=" * 50)
    print(f"Running Variance Threshold = {threshold}")
    print("=" * 50)

    start_time = time.time()

    # Feature selection
    selector = VarianceThreshold(threshold=threshold)
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train_selected.shape[1]

    print(f"Original features:  {X.shape[1]}")
    print(f"Selected features:  {selected_features}")

    # Scale reduced features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Train classifier
    print("Starting classifier training...")
    train_start = time.time()

    model = SGDClassifier(loss="log_loss", random_state=RANDOM_STATE)
    classes = y_train.unique()

    for epoch in range(50):
        model.partial_fit(X_train_scaled, y_train, classes=classes)

    train_time = time.time() - train_start
    print(f"Training finished in {train_time:.2f} seconds")

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    runtime = time.time() - start_time

    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {f1_macro:.4f}")
    print(f"Weighted F1:   {f1_weighted:.4f}")
    print(f"Runtime (s):   {runtime:.2f}")

    all_results.append({
        "Method": "Variance Threshold",
        "Threshold": threshold,
        "Original Features": X.shape[1],
        "Selected Features": selected_features,
        "Accuracy": acc,
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "Runtime (s)": runtime
    })

# -----------------------------
# STEP 3: Save Results Table
# -----------------------------
results = pd.DataFrame(all_results)
results.to_csv(f"{OUT_DIR}/variance_filter_results.csv", index=False)

# -----------------------------
# STEP 4: Save Threshold Comparison Plot
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(results["Threshold"], results["Accuracy"],
         marker="o", linewidth=2, label="Accuracy")
plt.plot(results["Threshold"], results["Macro F1"],
         marker="o", linewidth=2, label="Macro F1")
plt.plot(results["Threshold"], results["Weighted F1"],
         marker="o", linewidth=2, label="Weighted F1")

plt.title(f"{CANCER} - Variance Threshold Sweep")
plt.xlabel("Variance Threshold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/variance_threshold_sweep.png", dpi=150)
plt.close()

# -----------------------------
# STEP 5: Save Feature Count Plot
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(results["Threshold"], results["Selected Features"],
         marker="o", linewidth=2)

plt.title(f"{CANCER} - Selected Features vs Variance Threshold")
plt.xlabel("Variance Threshold")
plt.ylabel("Selected Features")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/variance_selected_features.png", dpi=150)
plt.close()

# -----------------------------
# STEP 6: Save Summary TXT
# -----------------------------
with open(f"{OUT_DIR}/variance_results.txt", "w") as f:
    f.write(f"=== {CANCER} Variance Threshold Sweep ===\n\n")
    f.write(results.to_string(index=False))
    f.write("\n\nBest Accuracy Threshold: ")
    f.write(str(results.loc[results["Accuracy"].idxmax(), "Threshold"]))
    f.write("\nBest Macro F1 Threshold: ")
    f.write(str(results.loc[results["Macro F1"].idxmax(), "Threshold"]))
    f.write("\nFastest Runtime Threshold: ")
    f.write(str(results.loc[results["Runtime (s)"].idxmin(), "Threshold"]))

print(f"\n[{CANCER}] Done - variance threshold sweep saved to {OUT_DIR}/")