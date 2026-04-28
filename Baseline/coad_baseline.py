import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

OUT_DIR = "H:/MLFinalProj/4824BRCAClassification/Baseline/results"
import os
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "COAD"
RANDOM_STATE = 42

# 1. Load expression matrix
print(f"[{CANCER}] Loading expression matrix...")
expr = pd.read_csv(
    f"H:/MLFinalProj/4824BRCAClassification/data/TCGA-{CANCER}/coad_expression_matrix.csv",
    index_col=0
)
print(f"  Expression shape: {expr.shape}")

# 2. Load subtypes
sub = pd.read_csv(
    f"H:/MLFinalProj/4824BRCAClassification/data/TCGA-{CANCER}/coad_subtypes.csv"
)
sub.columns = sub.columns.str.strip()
sub = sub[["pan.samplesID", "Subtype_Selected"]].dropna()
sub["sample_id"] = sub["pan.samplesID"].str[:12]

# 3. Align sample IDs (deduplicate by averaging)
expr.index = expr.index.str[:12]
expr = expr.groupby(expr.index).mean()
common = expr.index.intersection(sub["sample_id"])
print(f"  Common samples: {len(common)}")

X = expr.loc[common].values
y = sub.set_index("sample_id").loc[common, "Subtype_Selected"]

print(f"  Class distribution:\n{y.value_counts()}")

# 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

# 5. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train Logistic Regression
print(f"\n[{CANCER}] Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"  Accuracy:      {acc:.4f}")
print(f"  Macro F1:      {f1_macro:.4f}")
print(f"  Weighted F1:   {f1_weighted:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# 8. Confusion Matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"{CANCER} - Confusion Matrix (Logistic Regression Baseline)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/{CANCER}_confusion_matrix.png", dpi=150)
plt.close()
print(f"  Saved confusion matrix.")

# 9. Accuracy / F1 bar chart
plt.figure(figsize=(8, 5))
metrics = ["Accuracy", "Macro F1", "Weighted F1"]
values = [acc, f1_macro, f1_weighted]
colors = ["#4C72B0", "#55A868", "#C44E52"]
bars = plt.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.2)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
plt.ylim(0, 1.1)
plt.title(f"{CANCER} - Baseline Logistic Regression Performance")
plt.ylabel("Score")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/{CANCER}_metrics_bar.png", dpi=150)
plt.close()
print(f"  Saved metrics bar chart.")

# 10. Save results summary
with open(f"{OUT_DIR}/{CANCER}_results.txt", "w") as f:
    f.write(f"=== {CANCER} Baseline Logistic Regression ===\n\n")
    f.write(f"Accuracy:      {acc:.4f}\n")
    f.write(f"Macro F1:      {f1_macro:.4f}\n")
    f.write(f"Weighted F1:   {f1_weighted:.4f}\n\n")
    f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
    f.write(f"\nClass distribution (train):\n{y_train.value_counts().to_string()}\n")
    f.write(f"\nClass distribution (test):\n{y_test.value_counts().to_string()}\n")

print(f"[{CANCER}] Done - results saved to {OUT_DIR}/")
