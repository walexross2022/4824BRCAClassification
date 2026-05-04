# Filter/anova_filter.py

import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app import X, y

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Filter", "results")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "BRCA"
RANDOM_STATE = 42
TOP_K = 5000

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
# STEP 2: ANOVA Feature Selection
# -----------------------------
start_time = time.time()

selector = SelectKBest(score_func=f_classif, k=TOP_K)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train_selected.shape[1]

print(f"\nANOVA / F-test Feature Selection")
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
    print(f"Epoch {epoch + 1}/50 complete")

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
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# -----------------------------
# STEP 6: Confusion Matrix
# -----------------------------
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title(f"{CANCER} - ANOVA Filter Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/anova_confusion_matrix.png", dpi=150)
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
plt.title(f"{CANCER} - ANOVA Filter Performance")
plt.ylabel("Score")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/anova_metrics_bar.png", dpi=150)
plt.close()

# -----------------------------
# STEP 8: Save Results
# -----------------------------
results = pd.DataFrame([{
    "Method": "ANOVA / F-test",
    "Top K": TOP_K,
    "Original Features": X.shape[1],
    "Selected Features": selected_features,
    "Accuracy": acc,
    "Macro F1": f1_macro,
    "Weighted F1": f1_weighted,
    "Runtime (s)": runtime
}])

results.to_csv(f"{OUT_DIR}/anova_filter_results.csv", index=False)

with open(f"{OUT_DIR}/anova_results.txt", "w") as f:
    f.write(f"=== {CANCER} ANOVA / F-test Filter ===\n\n")
    f.write(f"Top K:         {TOP_K}\n")
    f.write(f"Accuracy:      {acc:.4f}\n")
    f.write(f"Macro F1:      {f1_macro:.4f}\n")
    f.write(f"Weighted F1:   {f1_weighted:.4f}\n")
    f.write(f"Runtime (s):   {runtime:.2f}\n\n")
    f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

print(f"\n[{CANCER}] Done - results saved to {OUT_DIR}/")